"""Register the G1 mjlab tasks used by the published checkpoints.

Importing this package registers the training tasks along the release
lineage and the isolated evaluation tasks. The 10 cm stair tasks in
``stairs10`` are registered on demand by the scripts that use them.
"""

from . import flat_deploy_no_cmd_curriculum_symloss as _flat_symloss
from . import mix_deploy_no_cmd_curriculum_symloss_yaww4 as _mix_yaww4
from . import mix3_deploy_no_cmd_curriculum_symloss_yaww4 as _mix3
from . import mix4_stairs5_deploy_no_cmd_curriculum_symloss_yaww4 as _mix4_stairs5
from . import slope_deploy_no_cmd_curriculum_symloss_yaww4 as _slope
from . import slopeinv_deploy_no_cmd_curriculum_symloss_yaww4 as _slopeinv
from . import stage2_gap_eval_tasks as _stage2_gap_eval
from . import stage4_stairs5_eval_tasks as _stage4_stairs5_eval

# Flat command tracking (G1-Flat-Deploy-NoCmdCurriculum-SymLoss).
_flat_symloss.register()
# Mixed-terrain training lineage of the release checkpoint.
_mix_yaww4.register()
_mix3.register()
# Mixed terrain with 0-5 cm stairs; starting point of the 10 cm experiment.
_mix4_stairs5.register()
# Isolated surfaces and continuous routes used for evaluation.
_slope.register()
_slopeinv.register()
_stage4_stairs5_eval.register()
_stage2_gap_eval.register()
