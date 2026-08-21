using UnrealBuildTool;
using System.Collections.Generic;

public class MT03_RealTimeLayoutEditorTarget : TargetRules
{
	public MT03_RealTimeLayoutEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("MT03_RealTimeLayout");
	}
}
