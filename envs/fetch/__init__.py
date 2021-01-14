from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .custom import CustomGoalEnv
from .extensions_interval import IntervalExt, IntervalColl, IntervalRewSub, IntervalRewVec, IntervalCollStopRegion,\
	IntervalCollStop, IntervalRewMod, IntervalRewModStop, IntervalRewModRegion, IntervalRewModRegionStop, \
	IntervalTestExtendedBbox, IntervalCollMinDist, IntervalMinDistRewMod,IntervalMinDistRewModStop, \
	IntervalTestExtendedMinDist,IntervalPAV,IntervalPAVRewMod,IntervalPAVRewModStop,IntervalTestExtendedPAV,\
	IntervalPAVRel,IntervalPAVRelRewMod,IntervalPAVRelRewModStop,IntervalTestExtendedPAVRel, IntervalTest

# TODO: change this file for new env handling!
def make_env(args):
	return {
		'vanilla': VanillaGoalEnv,
		'fixobj': FixedObjectGoalEnv,
		'interval': IntervalGoalEnv,
		'custom': CustomGoalEnv,
		'intervalTest':IntervalTest,
		'intervalExt':IntervalExt,
		'intervalColl':IntervalColl,
		'intervalRewSub':IntervalRewSub,
		'intervalRewVec': IntervalRewVec,
		'intervalTestExtendedBbox':IntervalTestExtendedBbox,
		'intervalCollStop':IntervalCollStop,
		'intervalRewMod': IntervalRewMod,#from this on, it miss variation where collsion gives -1 but not stop
		'intervalCollStopRegion': IntervalCollStopRegion,
		'intervalRewModStop': IntervalRewModStop,
		'intervalRewModRegion': IntervalRewModRegion,
		'intervalRewModRegionStop': IntervalRewModRegionStop,
		'intervalCollMinDist':IntervalCollMinDist,
		'intervalMinDistRewMod':IntervalMinDistRewMod,
		'intervalMinDistRewModStop':IntervalMinDistRewModStop,
		'intervalTestExtendedMinDist': IntervalTestExtendedMinDist,
		'intervalPAV':IntervalPAV,
		'intervalPAVRewMod':IntervalPAVRewMod,
		'intervalPAVRewModStop':IntervalPAVRewModStop,
		'intervalTestExtendedPAV':IntervalTestExtendedPAV,
		'intervalPAVRel':IntervalPAVRel,
		'intervalPAVRelRewMod':IntervalPAVRelRewMod,
		'intervalPAVRelRewModStop':IntervalPAVRelRewModStop,
		'intervalTestExtendedPAVRel':IntervalTestExtendedPAVRel
		
	}[args.goal](args)





def make_temp_env(args):
	return IntervalGoalEnv(args=args)
