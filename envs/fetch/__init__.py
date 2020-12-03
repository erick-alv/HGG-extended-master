from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .custom import CustomGoalEnv
from .interval_with_contact_detection import IntervalWithCollisionDetection
from .extensions_interval import IntervalExt, IntervalColl, IntervalRewSub, IntervalRewVec, IntervalCollStopRegion,\
	IntervalCollStop, IntervalRewMod, IntervalRewModStop, IntervalRewModRegion, IntervalRewModRegionStop, \
	IntervalTestExtendedBbox, IntervalCollMinDist, IntervalMinDistRewMod,IntervalMinDistRewModStop, \
	IntervalTestExtendedMinDist, IntervalPA,IntervalPARewMod,IntervalPARewModStop,IntervalTestExtendedPA,IntervalPAV,\
	IntervalPAVRewMod,IntervalPAVRewModStop,IntervalTestExtendedPAV,IntervalPARel,IntervalPARelRewMod,\
	IntervalPARelRewModStop,IntervalTestExtendedPARel,IntervalPAVRel,IntervalPAVRelRewMod,\
	IntervalPAVRelRewModStop,IntervalTestExtendedPAVRel

# TODO: change this file for new env handling!
def make_env(args):
	return {
		'vanilla': VanillaGoalEnv,
		'fixobj': FixedObjectGoalEnv,
		'interval': IntervalGoalEnv,
		'custom': CustomGoalEnv,
		'intervalCollision':IntervalWithCollisionDetection,
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
		'intervalPA': IntervalPA,
		'intervalPARewMod':IntervalPARewMod,
		'intervalPARewModStop':IntervalPARewModStop,
		'intervalTestExtendedPA':IntervalTestExtendedPA,
		'intervalPAV':IntervalPAV,
		'intervalPAVRewMod':IntervalPAVRewMod,
		'intervalPAVRewModStop':IntervalPAVRewModStop,
		'intervalTestExtendedPAV':IntervalTestExtendedPAV,
		'intervalPARel':IntervalPARel,
		'intervalPARelRewMod':IntervalPARelRewMod,
		'intervalPARelRewModStop':IntervalPARelRewModStop,
		'intervalTestExtendedPARel':IntervalTestExtendedPARel,
		'intervalPAVRel':IntervalPAVRel,
		'intervalPAVRelRewMod':IntervalPAVRelRewMod,
		'intervalPAVRelRewModStop':IntervalPAVRelRewModStop,
		'intervalTestExtendedPAVRel':IntervalTestExtendedPAVRel
		
	}[args.goal](args)



def make_temp_env(args):
	return IntervalGoalEnv(args=args)
