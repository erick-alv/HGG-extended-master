from .ddpg import DDPG
from .sac import SAC

def create_agent(args):
	return {
		'ddpg': DDPG,
		'sac':SAC
	}[args.alg](args)