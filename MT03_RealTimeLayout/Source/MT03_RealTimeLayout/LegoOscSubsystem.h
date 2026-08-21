// Auto-starting OSC receiver that mirrors every tracked Lego piece from
// lego_locator_xyz.py as a real extruded-outline mesh, built at runtime via
// GeometryScript - not a placeholder block. See docs/ROADMAP.md ("Real
// outline instead of a placeholder block") and
// tasks/2026-08-21-unreal-outline-extrude.md for the full background.
//
// This is a UWorldSubsystem, so it needs NO manual Blueprint/actor
// placement in the level - Unreal instantiates one automatically for every
// World the moment it starts (PIE or packaged), which is what "runs
// automatically" means here.
//
// NOT compiled/verified against a live Unreal Editor - written without
// engine access. The OSC plugin's delegate signature and a couple of
// UOSCManager/GeometryScript symbol names are the most likely things to
// need a small fix on first compile; see the inline comments at each call
// site for what to check in the 5.8 API reference if so.
#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "LegoOscSubsystem.generated.h"

class UOSCServer;
class ADynamicMeshActor;
struct FOSCMessage;

USTRUCT()
struct FLegoPieceState
{
	GENERATED_BODY()

	UPROPERTY()
	TObjectPtr<ADynamicMeshActor> Actor = nullptr;

	double LastSeenSeconds = 0.0;
};

/**
 * Wire schema this listens for (see README.md "OSC message layout"):
 *
 *   /obj     -> [name, x_mm, y_mm, angle_deg, w_cm, h_cm, z_mm, shape]
 *   /outline -> [name, n_points, x1_mm, y1_mm, ..., xn_mm, yn_mm, height_cm]
 *
 * /outline's vertices are already absolute floor-frame WORLD-space
 * coordinates and already carry the piece's true rotation (see
 * lego_locator_xyz.py's build_outline_mm()) - so each piece's mesh is built
 * directly in world space and its actor's own transform is left at
 * identity. This sidesteps double-rotating/double-positioning the mesh
 * once from /outline's baked-in transform and again from /obj's x/y/angle.
 *
 * height_cm is a FIXED extrusion height, the same for every piece
 * (--outline-height on lego_locator_xyz.py, default 2cm) - by design, not a
 * real per-piece scan, so every piece gets the same "thickness"/volume
 * envelope and the outline's 2D footprint is what actually carries the
 * shape information. /obj's `shape` field (index 7) is a coarser
 * square/rectangle/circle/cross category, kept only as a cheap fallback
 * signal - it is NOT used to pick a mesh here, since /outline's real
 * silhouette is always the more accurate source once it's arrived.
 */
UCLASS()
class MT03_REALTIMELAYOUT_API ULegoOscSubsystem : public UTickableWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override;

	/** Must match lego_locator_xyz.py's --osc-host (see locator_config.py's
	 * DEFAULTS / README.md). */
	UPROPERTY(EditAnywhere, Config, Category = "Lego OSC")
	FString ReceiveAddress = TEXT("127.0.0.1");

	/** Must match lego_locator_xyz.py's --osc-port. */
	UPROPERTY(EditAnywhere, Config, Category = "Lego OSC")
	int32 ReceivePort = 7000;

	/** A piece with no /outline update for this long is hidden - covers a
	 * piece being lifted off the table or the tracker losing it (see
	 * docs/PRODUCTION_READINESS.md #1/#2 for the matching Python-side gaps:
	 * there's currently no explicit "piece removed" message, so staleness
	 * is inferred here instead). */
	UPROPERTY(EditAnywhere, Config, Category = "Lego OSC")
	float StaleTimeoutSeconds = 2.0f;

private:
	UPROPERTY()
	TObjectPtr<UOSCServer> Server = nullptr;

	// Not a UPROPERTY (TMap<FString, FLegoPieceState> containing a struct
	// with a UPROPERTY member is fine to GC-track via the struct itself,
	// but the actors are also referenced by the map value - kept simple by
	// letting FLegoPieceState::Actor be the sole tracked reference).
	TMap<FString, FLegoPieceState> Pieces;

	UFUNCTION()
	void HandleOscMessage(const FOSCMessage& Message, const FString& SenderIPAddress, int32 SenderPort);

	void HandleOutlineMessage(const FOSCMessage& Message);
	void HandleObjMessage(const FOSCMessage& Message);

	ADynamicMeshActor* GetOrCreatePieceActor(const FString& Name);
};
