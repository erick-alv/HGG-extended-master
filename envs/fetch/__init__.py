from .vanilla import VanillaGoalEnv
from .fixobj import FixedObjectGoalEnv
from .interval import IntervalGoalEnv
from .custom import CustomGoalEnv
from .interval_with_contact_detection import IntervalWithCollisionDetection
from .extensions_interval import IntervalExt, IntervalColl, IntervalRewSub, IntervalRewVec, IntervalTestCollDetRewSub,\
	IntervalTestCollDetRewVec, IntervalEnvCollStop, IntervalSelfCollStop, IntervalTestColl, IntervalRewMod, \
	IntervalSelfCollStopRegion, IntervalRewModStop, IntervalRewModRegion,IntervalRewModRegionStop



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
		'intervalTestColl':IntervalTestColl,
		'intervalTestCollDetRewSub':IntervalTestCollDetRewSub,
		'intervalTestCollDetRewVec':IntervalTestCollDetRewVec,
		'intervalEnvCollStop':IntervalEnvCollStop,
		'intervalSelfCollStop':IntervalSelfCollStop,
		'intervalRewMod': IntervalRewMod,#from this on, it miss varation where collsion gives -1 but not stop
		'intervalSelfCollStopRegion': IntervalSelfCollStopRegion,
		'intervalRewModStop': IntervalRewModStop,
		'intervalRewModRegion': IntervalRewModRegion,
		'intervalRewModRegionStop': IntervalRewModRegionStop,

	}[args.goal](args)

def make_temp_env(args):
	return IntervalGoalEnv(args=args)