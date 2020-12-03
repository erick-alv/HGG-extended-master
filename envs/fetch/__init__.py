from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .custom import CustomGoalEnv
from .interval_with_contact_detection import IntervalWithCollisionDetection
from .extensions_interval import IntervalExt, IntervalColl, IntervalRewSub, IntervalRewVec, IntervalCollStopRegion,\
	IntervalCollStop, IntervalRewMod, IntervalRewModStop, IntervalRewModRegion, IntervalRewModRegionStop, \
	IntervalTestExtendedBbox, IntervalCollMinDist, IntervalMinDistRewMod,IntervalMinDistRewModStop, \
	IntervalTestExtendedMinDist

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
		'intervalTestExtendedMinDist': IntervalTestExtendedMinDist
	}[args.goal](args)


def make_temp_env(args):
	return IntervalGoalEnv(args=args)
