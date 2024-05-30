from helpers.annotations import *

## DO NOT INSTANTIATE THIS CLASS! USE THE OBJECT IN THE GLOBAL UTILS!!

class Constants(object):
    
    @constant
    def PATH_TO_CHECKPOINT_LLMMREC_MOVIES():
        return './checkpoints/llmrec/movies' 
    
    @constant
    def PATH_TO_CHECKPOINT_LLMMREC_TOYS():
        return './checkpoints/llmrec/toys' 