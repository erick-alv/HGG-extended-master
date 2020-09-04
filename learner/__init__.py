from .hgg import HGGLearner, HGGLearner_VAEs, NormalLearner

learner_collection = {
	'normal': NormalLearner,
	'hgg': HGGLearner,
	'hgg_with_VAES': HGGLearner_VAEs
}

def create_learner(args):
	if args.vae_dist_help:
		return HGGLearner_VAEs(args)
	return learner_collection[args.learn](args)