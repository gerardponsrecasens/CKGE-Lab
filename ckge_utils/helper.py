import os
import torch
import torch.nn as nn
from pykeen.triples import TriplesFactory
from pykeen.nn.init import PretrainedInitializer
from collections import defaultdict
from .expand_models import *

def switch_snapshot(model, entity_to_id, relation_to_id, model_name, snapshot, dataset_name, initialization = 'random', initialization_args={}, inverse_triples = False):
    """
    Expand a PyKEEN model to include new entities/relations from a new snapshot.

    Args:
        model: PyKEEN model (e.g., TransE) trained on the previous snapshot.
        entity_to_id (dict): mapping from entity name to ID.
        relation_to_id (dict): mapping from relation name to ID.
        model_name (str): model type, e.g. 'TransE'.
        snapshot (int): snapshot index to load (e.g., 1).
        dataset_name (str): dataset folder name under ./data/
        initialization (str): which initialization apply to new entities/relations. By default, random applies. Available= 'random', 'schema'
        initialization_args (dict): args depending on the initilization
            -schema initialization: 
                -class_dict (dict): a dictionary which has for each entity name, the classes it belongs to e.g., {'ent_1':[class1,class2],...}
                -random_noise (float): indicates the amount of random perturbation applied to the initialization

    Returns:
        (model, entity_to_id, relation_to_id): expanded model and updated mappings.
    """
    
    snapshot_dir = f"data/{dataset_name}/{snapshot}"
    train_path = os.path.join(snapshot_dir, "train.txt")

    # Temporarily load triples from new snapshot
    new_tf_tmp = TriplesFactory.from_path(train_path, create_triples_factory=inverse_triples)

    # Detect new entities / relations
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_tf_tmp.entity_to_id.keys()) - old_entities
    new_relations = set(new_tf_tmp.relation_to_id.keys()) - old_relations


    # Create updated mappings
    new_entity_to_id = dict(entity_to_id)  
    new_relation_to_id = dict(relation_to_id)

    next_entity_id = max(entity_to_id.values()) + 1 if entity_to_id else 0
    next_relation_id = max(relation_to_id.values()) + 1 if relation_to_id else 0

    for e in new_entities:
        new_entity_to_id[e] = next_entity_id
        next_entity_id += 1

    for r in new_relations:
        new_relation_to_id[r] = next_relation_id
        next_relation_id += 1
    
    expand_fns = {
    "transe": expand_TransE,
    "transh": expand_TransH,
    "rotate": expand_RotatE,
    "transr": expand_TransR,
    "distmult": expand_DistMult,
    "hole": expand_HolE,
    "proje": expand_ProjE,
    "complex": expand_ComplEx,
    "rescal": expand_RESCAL,
    "transf": expand_TransF,
    "tucker": expand_TuckER,
    "conve": expand_ConvE,
    "simple": expand_SimplE,
    "boxe": expand_BoxE,
    "transd": expand_TransD,
    "toruse": expand_TorusE,
    "pairre": expand_PairRE,
    "cp": expand_CP,
    "mure": expand_MuRE,
    "quate": expand_QuatE,
    "crosse": expand_CrossE,
    "distma": expand_DistMA,
    "convkb": expand_ConvKB,
    "kg2e": expand_KG2E,
    "ermlp": expand_ERMLP,
}

    key = model_name.lower()
    try:
        expand_fn = expand_fns[key]
    except KeyError:
        raise NotImplementedError(f"Model {model_name} not yet supported in switch_snapshot")

    new_model = expand_fn(
        model,
        new_entity_to_id,
        entity_to_id,
        new_relation_to_id,
        relation_to_id,
        initialization,
        initialization_args,
        train_path,
    )

    

    return new_model, new_entity_to_id, new_relation_to_id




from pykeen.training import SLCWATrainingLoop

class CKGETrainingLoop(SLCWATrainingLoop):
    def _train(self, *args, **kwargs):
        # temporarily override model.reset_parameters_
        original_reset = getattr(self.model, "reset_parameters_", None)
        self.model.reset_parameters_ = lambda *a, **k: None
        try:
            return super()._train(*args, **kwargs)
        finally:
            if original_reset is not None:
                self.model.reset_parameters_ = original_reset



def create_triples_factory(dataset_name, snapshot, entity_to_id = None, relation_to_id = None, inverse_triples=False):
    """
    Create the three triples factory from a Continual KGE Dataset

    Args:
        dataset_name (str): name of the dataset located in ./data/
        snapshot (int): snapshot index to load (e.g., 1).
        entity_to_id (dict): mapping from entity name to ID.
        
    Returns:
        (train_tf, valid_tf, test_tf): TriplesFactories.
    """
    snapshot_dir = f"data/{dataset_name}/{snapshot}"

    if snapshot == 0:
        train_tf = TriplesFactory.from_path(os.path.join(snapshot_dir, "train.txt"), create_inverse_triples=inverse_triples)
        valid_tf = TriplesFactory.from_path(os.path.join(snapshot_dir, "valid.txt"),
                                            entity_to_id=train_tf.entity_to_id,
                                            relation_to_id=train_tf.relation_to_id,  create_inverse_triples=inverse_triples)
        test_tf = TriplesFactory.from_path(os.path.join(snapshot_dir, "test.txt"),
                                        entity_to_id=train_tf.entity_to_id,
                                        relation_to_id=train_tf.relation_to_id,  create_inverse_triples=inverse_triples)
    else:
        train_tf = TriplesFactory.from_path(
            os.path.join(snapshot_dir, "train.txt"),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
             create_inverse_triples=inverse_triples
        )
        valid_tf = TriplesFactory.from_path(
            os.path.join(snapshot_dir, "valid.txt"),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            create_inverse_triples=inverse_triples
        )
        test_tf = TriplesFactory.from_path(
            os.path.join(snapshot_dir, "test.txt"),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            create_inverse_triples=inverse_triples
        )
    
    return train_tf, valid_tf, test_tf



import pickle

def save_snapshot(model, train_tf, snapshot_dir):
    """
    Saves the trained embeddings and the mapping dictionaries after training a CKGE model in 
    a snapshot.

    Args:
        model (PyKEEN model): trained PyKEEN model
        train_tf (TripleFactory): TripleFactory used for training
        snapshot_dir (str): directory where to save the trained dir
    
    """
    save_dir = os.path.join(snapshot_dir, "trained_model")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pt"))
    with open(os.path.join(save_dir, "entity_to_id.pkl"), "wb") as f:
        pickle.dump(train_tf.entity_to_id, f)
    with open(os.path.join(save_dir, "relation_to_id.pkl"), "wb") as f:
        pickle.dump(train_tf.relation_to_id, f)



def get_metrics(number_snapshots, test_list, total_results):


    total_testing = 0
    for i in range(number_snapshots):
        total_testing += test_list[i].mapped_triples.shape[0]

    # Forgetting:
    cf_h3 = 0
    cf_mmr = 0
    for i in range(number_snapshots-1):
    
        initial = total_results[i][i]['tail']['realistic']['hits_at_3']
        final = total_results[number_snapshots-1][i]['tail']['realistic']['hits_at_3']
        if initial > 0:
            cf_h3 += final/initial*(test_list[i].mapped_triples.shape[0]/total_testing)
        else:
            cf_h3 +=0

        

        initial = total_results[i][i]['tail']['realistic']['inverse_harmonic_mean_rank']
        final = total_results[number_snapshots-1][i]['tail']['realistic']['inverse_harmonic_mean_rank']
        if initial >0:
            cf_mmr += final/initial*(test_list[i].mapped_triples.shape[0]/total_testing)
        else:
            cf_mrr +=0

    # Knowledge Aquisition
    new_h3 = 0
    new_mrr = 0
    for i in range(1,number_snapshots):
        new_h3 += total_results[i][i]['tail']['realistic']['hits_at_3']
        new_mrr += total_results[i][i]['tail']['realistic']['inverse_harmonic_mean_rank']
    new_h3 /= (number_snapshots-1)
    new_mrr /= (number_snapshots-1)

    # Metrics
    mrr, h1, h3, h10 = 0,0,0,0

    for i in range(number_snapshots):
        mrr += total_results[number_snapshots-1][i]['tail']['realistic']['inverse_harmonic_mean_rank']*(test_list[i].mapped_triples.shape[0]/total_testing)
        h1 += total_results[number_snapshots-1][i]['tail']['realistic']['hits_at_1']*(test_list[i].mapped_triples.shape[0]/total_testing)
        h3 += total_results[number_snapshots-1][i]['tail']['realistic']['hits_at_3']*(test_list[i].mapped_triples.shape[0]/total_testing)
        h10 += total_results[number_snapshots-1][i]['tail']['realistic']['hits_at_10']*(test_list[i].mapped_triples.shape[0]/total_testing)

    metrics = {'cf_mrr':cf_mmr,'cf_h3':cf_h3, 'new_mrr':new_mrr, 'new_cf':cf_h3, 'mrr':mrr, 'hits@1':h1, 'hits@3':h3, 'hits@10':h10}


    return metrics