using UnrealBuildTool;
using System.Collections.Generic;

public class MT03_RealTimeLayoutTarget : TargetRules
{
	public MT03_RealTimeLayoutTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("MT03_RealTimeLayout");
	}
}
