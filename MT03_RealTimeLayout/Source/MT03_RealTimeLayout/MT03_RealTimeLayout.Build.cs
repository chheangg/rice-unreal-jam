using UnrealBuildTool;

public class MT03_RealTimeLayout : ModuleRules
{
	public MT03_RealTimeLayout(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"OSC",
			"GeometryScriptingCore",
			"GeometryFramework",
			"GeometryCore",
		});

		PrivateDependencyModuleNames.AddRange(new string[] { });
	}
}
