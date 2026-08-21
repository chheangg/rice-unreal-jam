#include "LegoOscSubsystem.h"

#include "OSCManager.h"
#include "OSCServer.h"
#include "OSCMessage.h"
#include "OSCAddress.h"

#include "GeometryScript/MeshPrimitiveFunctions.h"
#include "Components/DynamicMeshComponent.h"
#include "DynamicMeshActor.h"
#include "UDynamicMesh.h"

#include "Engine/World.h"

void ULegoOscSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	// CHECK IF THIS DOESN'T COMPILE: UOSCManager::CreateOSCServer's exact
	// parameter order/names have shifted across engine versions - if 5.8
	// differs, the node's Blueprint equivalent is "Create OSC Server" in
	// the OSC plugin; match this call's args to that node's pins.
	Server = UOSCManager::CreateOSCServer(
		ReceiveAddress,
		ReceivePort,
		/*bMulticastLoopback*/ false,
		/*bStartListening*/ true,
		TEXT("LegoOscServer"),
		this);

	if (Server)
	{
		// CHECK IF THIS DOESN'T COMPILE: delegate name/signature on
		// UOSCServer - Blueprint equivalent is the "On Osc Message
		// Received" event node on the OSC Server object.
		Server->OnOscMessageReceived.AddDynamic(this, &ULegoOscSubsystem::HandleOscMessage);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("LegoOscSubsystem: failed to start OSC server on %s:%d"),
			*ReceiveAddress, ReceivePort);
	}
}

void ULegoOscSubsystem::Deinitialize()
{
	if (Server)
	{
		Server->Stop();
		Server = nullptr;
	}
	Pieces.Empty();
	Super::Deinitialize();
}

void ULegoOscSubsystem::Tick(float DeltaTime)
{
	const double Now = FPlatformTime::Seconds();
	for (auto& Pair : Pieces)
	{
		FLegoPieceState& State = Pair.Value;
		if (!State.Actor)
		{
			continue;
		}
		const bool bStale = (Now - State.LastSeenSeconds) > StaleTimeoutSeconds;
		if (State.Actor->IsHidden() != bStale)
		{
			State.Actor->SetActorHiddenInGame(bStale);
		}
	}
}

TStatId ULegoOscSubsystem::GetStatId() const
{
	RETURN_QUICK_DECLARE_CYCLE_STAT(ULegoOscSubsystem, STATGROUP_Tickables);
}

void ULegoOscSubsystem::HandleOscMessage(const FOSCMessage& Message, const FString& SenderIPAddress, int32 SenderPort)
{
	// CHECK IF THIS DOESN'T COMPILE: FOSCAddress's "full path as string"
	// accessor - Blueprint equivalent is "Get OSC Address (as String)" /
	// UOSCManager::GetOSCAddressString on the message's address.
	const FString AddrString = Message.GetAddress().GetFullPath();

	if (AddrString == TEXT("/outline"))
	{
		HandleOutlineMessage(Message);
	}
	else if (AddrString == TEXT("/obj"))
	{
		HandleObjMessage(Message);
	}
}

void ULegoOscSubsystem::HandleOutlineMessage(const FOSCMessage& Message)
{
	// [name, n_points, x1_mm, y1_mm, ..., xn_mm, yn_mm, height_cm]
	// See lego_locator_xyz.py's build_outline_mm()/main() and README.md.
	FString Name;
	if (!UOSCManager::GetStringArgAt(Message, 0, Name))
	{
		return;
	}

	int32 NumPoints = 0;
	if (!UOSCManager::GetInt32ArgAt(Message, 1, NumPoints) || NumPoints < 3)
	{
		// lego_locator_xyz.py never sends n_points < 3 (see
		// docs/PRODUCTION_READINESS.md #? / the degenerate-polygon fix in
		// build_outline_mm()), but stay defensive against a malformed or
		// out-of-sync sender.
		return;
	}

	TArray<FVector2D> PolygonVerts;
	PolygonVerts.Reserve(NumPoints);
	int32 ArgIndex = 2;
	bool bOk = true;
	for (int32 i = 0; i < NumPoints && bOk; ++i)
	{
		float XMm = 0.f, YMm = 0.f;
		bOk = bOk && UOSCManager::GetFloatArgAt(Message, ArgIndex++, XMm);
		bOk = bOk && UOSCManager::GetFloatArgAt(Message, ArgIndex++, YMm);
		// mm -> Unreal units (1uu = 1cm): divide by 10. Do NOT use the
		// unrelated pixel-era "/50" divisor mentioned for detect.py in
		// docs/ROADMAP.md - that's for a different, non-metric pipeline.
		PolygonVerts.Add(FVector2D(XMm / 10.0, YMm / 10.0));
	}
	if (!bOk)
	{
		return;
	}

	float HeightCm = 2.0f;
	if (!UOSCManager::GetFloatArgAt(Message, ArgIndex, HeightCm))
	{
		return;
	}

	ADynamicMeshActor* PieceActor = GetOrCreatePieceActor(Name);
	if (!PieceActor)
	{
		return;
	}

	UDynamicMeshComponent* MeshComp = PieceActor->GetDynamicMeshComponent();
	UDynamicMesh* Mesh = MeshComp->GetDynamicMesh();
	Mesh->Reset();

	// CHECK IF THIS DOESN'T COMPILE: AppendExtrudedPolygon's exact
	// parameter list - Blueprint equivalent is the "Append Extruded
	// Polygon" node in the GeometryScript mesh-primitive function library.
	// height_cm is used directly as the extrude height, both in real-world
	// centimeters and Unreal's 1uu=1cm world unit - no conversion needed.
	FGeometryScriptPrimitiveOptions PrimitiveOptions;
	UGeometryScriptLibrary_MeshPrimitiveFunctions::AppendExtrudedPolygon(
		Mesh,
		PrimitiveOptions,
		FTransform::Identity, // vertices are already absolute world-space
		PolygonVerts,
		HeightCm);

	MeshComp->NotifyMeshUpdated();

	FLegoPieceState& State = Pieces.FindOrAdd(Name);
	State.Actor = PieceActor;
	State.LastSeenSeconds = FPlatformTime::Seconds();
	if (PieceActor->IsHidden())
	{
		PieceActor->SetActorHiddenInGame(false);
	}
}

void ULegoOscSubsystem::HandleObjMessage(const FOSCMessage& Message)
{
	FString Name;
	if (!UOSCManager::GetStringArgAt(Message, 0, Name))
	{
		return;
	}

	// /outline's vertices already carry the piece's true world-space
	// position and rotation (see class comment) - re-applying /obj's x/y/
	// angle to the actor's own transform on top of that would
	// double-transform the mesh, so /obj is only used here as a staleness
	// heartbeat between /outline updates. Its `shape` field (index 7) is
	// available if a future fallback path wants it (e.g. a debug label, or
	// spawning a primitive before the first /outline arrives).
	if (FLegoPieceState* Existing = Pieces.Find(Name))
	{
		Existing->LastSeenSeconds = FPlatformTime::Seconds();
	}
}

ADynamicMeshActor* ULegoOscSubsystem::GetOrCreatePieceActor(const FString& Name)
{
	if (FLegoPieceState* Existing = Pieces.Find(Name))
	{
		if (Existing->Actor)
		{
			return Existing->Actor;
		}
	}

	UWorld* World = GetWorld();
	if (!World)
	{
		return nullptr;
	}

	FActorSpawnParameters SpawnParams;
	SpawnParams.Name = MakeUniqueObjectName(World, ADynamicMeshActor::StaticClass(),
		FName(*FString::Printf(TEXT("LegoPiece_%s"), *Name)));
	SpawnParams.NameMode = FActorSpawnParameters::ESpawnActorNameMode::Requested;

	ADynamicMeshActor* NewActor = World->SpawnActor<ADynamicMeshActor>(
		FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);
	if (NewActor)
	{
		NewActor->GetDynamicMeshComponent()->SetMobility(EComponentMobility::Movable);
	}

	FLegoPieceState& State = Pieces.FindOrAdd(Name);
	State.Actor = NewActor;
	return NewActor;
}
