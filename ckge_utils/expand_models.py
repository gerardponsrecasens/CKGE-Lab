import torch
import torch.nn as nn
from pykeen.triples import TriplesFactory
from pykeen.models import *
from pykeen.nn.init import PretrainedInitializer
from collections import defaultdict
import numpy as np
from pykeen.nn.init import init_quaternions, xavier_uniform_, xavier_normal_, uniform_norm_, xavier_uniform_norm_, init_phases
from pykeen.utils import clamp_norm as clamp_norm_, complex_normalize

def initialize(method,tensor,initialization_args,old_entities, new_entities, new_entity_to_id):

    if method == 'random':
        return tensor
    
    if method == 'schema':

        class_dict = initialization_args['class_dict']
        random_noise = initialization_args['random_noise']

        class_to_entities = defaultdict(list)
        for entity in old_entities:
            idx = new_entity_to_id[entity]
            for c in class_dict[entity]:
                class_to_entities[c].append(idx)

        # Compute class averages and stds using tensor operations
        class_avg = {}
        class_std = {}

        for c, idx_list in class_to_entities.items():
            idx_tensor = torch.tensor(idx_list)
            embeddings = tensor[idx_tensor]  # Shape: [N, emb_dim]

            avg = embeddings.mean(dim=0, keepdim=True)  # Shape: [1, emb_dim]
            std = embeddings.std(dim=0, unbiased=False, keepdim=True)  # Shape: [1, emb_dim]

            class_avg[c] = avg
            class_std[c] = std
        
        
        # Initialize new entity embeddings based on class averages
        for ent in new_entities:
            idx = new_entity_to_id[ent]
            ent_classes = class_dict[ent]

            # Only consider classes with prior entity embeddings
            prev_classes = [c for c in ent_classes if c in class_avg]

            if prev_classes:
                avg_stack = torch.cat([class_avg[c] for c in prev_classes], dim=0)  # [K, emb_dim]
                std_stack = torch.cat([class_std[c] for c in prev_classes], dim=0)  # [K, emb_dim]

                mean_avg = avg_stack.mean(dim=0, keepdim=True)
                mean_std = std_stack.mean(dim=0, keepdim=True)

                noise = torch.randn_like(mean_avg) * mean_std * random_noise
                tensor[idx] = mean_avg + noise
        
        return tensor


def expand_TransE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.xavier_uniform_(expanded_ent_emb, gain=nn.init.calculate_gain("relu"))

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb.device)
    nn.init.xavier_uniform_(expanded_rel_emb, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]
    
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights
    new_model = TransE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)
        
    return new_model


def expand_TransH(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data.clone()
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.xavier_uniform_(expanded_ent_emb, gain=nn.init.calculate_gain("relu"))
    expanded_ent_emb = torch.nn.functional.normalize(expanded_ent_emb, p=2, dim=-1)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]


    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)
    
    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data.clone()
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb.device)
    nn.init.xavier_uniform_(expanded_rel_emb_0, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb[old_idx]
    

    # Relations
    old_rel_emb = model.relation_representations[1]._embeddings.weight.data.clone()
    expanded_rel_emb_1 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb.device)
    nn.init.xavier_uniform_(expanded_rel_emb_1, gain=nn.init.calculate_gain("relu"))
    expanded_rel_emb_1 = torch.nn.functional.normalize(expanded_rel_emb_1, p=2, dim=-1)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_1[new_relation_to_id[r]] = old_rel_emb[old_idx]
    
    
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'relation_representations.1._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}  


    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = TransH(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)
    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_rel_emb_1)



        
    return new_model


def expand_HolE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    expanded_ent_emb = xavier_uniform_(expanded_ent_emb)
    expanded_ent_emb = clamp_norm_(expanded_ent_emb, maxnorm=1, p=2, dim=-1)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    expanded_rel_emb = xavier_uniform_(expanded_rel_emb)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]


    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}


    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = HolE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model


def expand_RotatE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Determine dimensions
    emb_dim = emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]//2       
    full_dim = emb_dim * 2                                        

    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data  # [n_old, 2*emb_dim]
    expanded_ent_emb = torch.empty(len(new_entity_to_id), full_dim, device=old_ent_emb.device)
    nn.init.xavier_uniform_(expanded_ent_emb)

    # Copy old embeddings
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data  # [n_old, 2*emb_dim]
    expanded_rel_emb = torch.empty(len(new_relation_to_id), full_dim, device=old_rel_emb.device)
    expanded_rel_emb = init_phases(expanded_rel_emb)
    expanded_rel_emb = expanded_rel_emb.permute(0, 2, 1).reshape(expanded_rel_emb.size(0), -1)
    expanded_rel_emb = complex_normalize(expanded_rel_emb)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]
    
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}


    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )


    new_model = RotatE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed=model._random_seed,
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

    return new_model


def expand_TransR(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id,
                   initialization, initialization_args, train_path):

    # --- Dimensions ---
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    proj_dim = model.relation_representations[1]._embeddings.weight.shape[1]  

    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    device = model.entity_representations[0]._embeddings.weight.device

    # Entities 
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data.clone()
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=device)
    nn.init.xavier_uniform_(expanded_ent_emb, gain=nn.init.calculate_gain("relu"))
    # Copy old
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]
    # Clamp norm for new entities
    if new_entities:
        new_idx = torch.tensor([new_entity_to_id[e] for e in new_entities], device=device)
        expanded_ent_emb[new_idx] = clamp_norm_(expanded_ent_emb[new_idx], maxnorm=1.0, p=2, dim=-1)

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)
    
    # Relations 
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data.clone()
    expanded_rel_emb = torch.empty(len(new_relation_to_id), rel_dim, device=device)
    nn.init.xavier_uniform_(expanded_rel_emb, gain=nn.init.calculate_gain("relu"))
    # Copy old
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]
    # Clamp norm for new relations
    if new_relations:
        new_idx = torch.tensor([new_relation_to_id[r] for r in new_relations], device=device)
        expanded_rel_emb[new_idx] = clamp_norm_(expanded_rel_emb[new_idx], maxnorm=1.0, p=2, dim=-1)

    # Relation projections
    old_proj = model.relation_representations[1]._embeddings.weight.data.clone()  # [num_rel, rel_dim*emb_dim]
    expanded_proj = torch.empty(len(new_relation_to_id), proj_dim, device=device)
    nn.init.xavier_uniform_(expanded_proj,gain=nn.init.calculate_gain("relu"))
    # Copy old projections
    for r, old_idx in relation_to_id.items():
        expanded_proj[new_relation_to_id[r]] = old_proj[old_idx]


    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'relation_representations.1._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}
    
    # TriplesFactory 
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Build new TransR model
    new_model = TransR(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        relation_dim=rel_dim,
        random_seed=model._random_seed,
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_proj)

    return new_model


def expand_DistMult(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):

    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.xavier_uniform_(expanded_ent_emb, gain=nn.init.calculate_gain("relu"))
    expanded_ent_emb = torch.nn.functional.normalize(expanded_ent_emb, p=2, dim=-1)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    nn.init.xavier_normal_(expanded_rel_emb, gain=nn.init.calculate_gain("relu"))
    expanded_rel_emb = torch.nn.functional.normalize(expanded_rel_emb, p=2, dim=-1)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}       

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = DistMult(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model

def expand_ProjE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    expanded_ent_emb = xavier_uniform_(expanded_ent_emb)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    expanded_rel_emb = xavier_uniform_(expanded_rel_emb)


    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]


        
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = ProjE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model


def expand_ComplEx(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):

    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.normal_(expanded_ent_emb)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    nn.init.normal_(expanded_rel_emb)
    
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]

    

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}



    new_model = ComplEx(
        triples_factory=new_tf,
        embedding_dim=emb_dim//2,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model

def expand_RESCAL(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.uniform_(expanded_ent_emb)


    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), emb_dim*emb_dim, device=old_rel_emb.device)
    nn.init.uniform_(expanded_rel_emb)
    
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}       


    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = RESCAL(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model


def expand_TransF(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):

    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.xavier_uniform_(expanded_ent_emb, gain=nn.init.calculate_gain("relu"))
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    nn.init.xavier_uniform_(expanded_rel_emb)

    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]


        

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = TransF(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model

def expand_TuckER(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    expanded_ent_emb = xavier_normal_(expanded_ent_emb)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb.device)
    expanded_rel_emb = xavier_normal_(expanded_rel_emb)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = TuckER(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model


def expand_ConvE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    inverse_triples = True
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    expanded_ent_emb_0 = xavier_normal_(expanded_ent_emb_0)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]
    
    # Entities
    old_ent_emb_1 = model.entity_representations[1]._embeddings.weight.data
    expanded_ent_emb_1 = torch.empty(len(new_entity_to_id), model.entity_representations[1]._embeddings.weight.shape[1], device=old_ent_emb_1.device)
    expanded_ent_emb_1 = xavier_normal_(expanded_ent_emb_1)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_1[new_entity_to_id[e]] = old_ent_emb_1[old_idx]

    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)
    expanded_ent_emb_1 = initialize(initialization,expanded_ent_emb_1,initialization_args,old_entities, new_entities, new_entity_to_id)
    
    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(model.relation_representations[0]._embeddings.weight.shape[0], rel_dim, device=old_rel_emb.device)
    expanded_rel_emb = xavier_normal_(expanded_rel_emb)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]
        if inverse_triples:
            expanded_rel_emb[new_relation_to_id[r]+len(new_relation_to_id.keys())] = old_rel_emb[old_idx+len(old_relations)]


        
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'entity_representations.1._embeddings.weight',
                        'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id, create_inverse_triples= inverse_triples
    )

    # Expand model inheriting weights

    new_model = ConvE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
        
    )


    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.entity_representations[1]._embeddings.weight.copy_(expanded_ent_emb_1)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model


def expand_SimplE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities 0
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    nn.init.xavier_uniform_(expanded_ent_emb_0, gain=nn.init.calculate_gain("relu"))
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]

    
    # Entities 1
    old_ent_emb_1 = model.entity_representations[1]._embeddings.weight.data
    expanded_ent_emb_1 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_1.device)
    nn.init.xavier_uniform_(expanded_ent_emb_1, gain=nn.init.calculate_gain("relu"))

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_1[new_entity_to_id[e]] = old_ent_emb_1[old_idx]


    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)
    expanded_ent_emb_1 = initialize(initialization,expanded_ent_emb_1,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations 0
    old_rel_emb_0 = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_0.device)
    nn.init.xavier_normal_(expanded_rel_emb_0, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb_0[old_idx]
    

    # Relations 1
    old_rel_emb_1 = model.relation_representations[1]._embeddings.weight.data
    expanded_rel_emb_1 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_1.device)
    nn.init.xavier_normal_(expanded_rel_emb_1, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_1[new_relation_to_id[r]] = old_rel_emb_1[old_idx]

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'entity_representations.1._embeddings.weight', 'relation_representations.1._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}


    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = SimplE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.entity_representations[1]._embeddings.weight.copy_(expanded_ent_emb_1)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_rel_emb_1)

    new_model.load_state_dict(filtered_state_dict, strict=False)
        
    return new_model


def expand_BoxE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities 0
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    expanded_ent_emb_0 = uniform_norm_(expanded_ent_emb_0)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]

    
    # Entities 1
    old_ent_emb_1 = model.entity_representations[1]._embeddings.weight.data
    expanded_ent_emb_1 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_1.device)
    expanded_ent_emb_1 = uniform_norm_(expanded_ent_emb_1)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_1[new_entity_to_id[e]] = old_ent_emb_1[old_idx]


    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)
    expanded_ent_emb_1 = initialize(initialization,expanded_ent_emb_1,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations 0
    old_rel_emb_0 = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_0.device)
    expanded_rel_emb_0 = uniform_norm_(expanded_rel_emb_0)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb_0[old_idx]
    

    # Relations 1
    old_rel_emb_1 = model.relation_representations[1]._embeddings.weight.data
    expanded_rel_emb_1 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_1.device)
    expanded_rel_emb_1 = uniform_norm_(expanded_rel_emb_1)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_1[new_relation_to_id[r]] = old_rel_emb_1[old_idx]

    # Relations 3
    old_rel_emb_3 = model.relation_representations[3]._embeddings.weight.data
    expanded_rel_emb_3 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_3.device)
    expanded_rel_emb_3 = uniform_norm_(expanded_rel_emb_3)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_3[new_relation_to_id[r]] = old_rel_emb_3[old_idx]
    
    # Relations 4
    old_rel_emb_4 = model.relation_representations[4]._embeddings.weight.data
    expanded_rel_emb_4 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_4.device)
    expanded_rel_emb_4 = uniform_norm_(expanded_rel_emb_4)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_4[new_relation_to_id[r]] = old_rel_emb_4[old_idx]


    # Relations 2
    old_rel_emb_2 = model.relation_representations[2]._embeddings.weight.data
    expanded_rel_emb_2 = torch.empty(len(new_relation_to_id), 1, device=old_rel_emb_2.device)
    expanded_rel_emb_2 = uniform_norm_(expanded_rel_emb_2)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_2[new_relation_to_id[r]] = old_rel_emb_2[old_idx]

    
    # Relations 5
    old_rel_emb_5 = model.relation_representations[5]._embeddings.weight.data
    expanded_rel_emb_5 = torch.empty(len(new_relation_to_id), 1, device=old_rel_emb_5.device)
    expanded_rel_emb_5 = uniform_norm_(expanded_rel_emb_5)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_5[new_relation_to_id[r]] = old_rel_emb_5[old_idx]
    

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'entity_representations.1._embeddings.weight', 'relation_representations.1._embeddings.weight',
                       'relation_representations.2._embeddings.weight','relation_representations.3._embeddings.weight',
                       'relation_representations.4._embeddings.weight', 'relation_representations.5._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}



    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = BoxE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.entity_representations[1]._embeddings.weight.copy_(expanded_ent_emb_1)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_rel_emb_1)
        new_model.relation_representations[2]._embeddings.weight.copy_(expanded_rel_emb_2)
        new_model.relation_representations[3]._embeddings.weight.copy_(expanded_rel_emb_3)
        new_model.relation_representations[4]._embeddings.weight.copy_(expanded_rel_emb_4)
        new_model.relation_representations[5]._embeddings.weight.copy_(expanded_rel_emb_5)

        
    new_model.load_state_dict(filtered_state_dict, strict=False)

    return new_model


def expand_TransD(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities 0
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    expanded_ent_emb_0 = xavier_uniform_(expanded_ent_emb_0)
    expanded_ent_emb_0 = clamp_norm_(expanded_ent_emb_0,maxnorm=1, p=2, dim=-1)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]

    
    # Entities 1
    old_ent_emb_1 = model.entity_representations[1]._embeddings.weight.data
    expanded_ent_emb_1 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_1.device)
    expanded_ent_emb_1 = xavier_uniform_(expanded_ent_emb_1)
    expanded_ent_emb_1 = clamp_norm_(expanded_ent_emb_1,maxnorm=1, p=2, dim=-1)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_1[new_entity_to_id[e]] = old_ent_emb_1[old_idx]

    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)
    expanded_ent_emb_1 = initialize(initialization,expanded_ent_emb_1,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations 0
    old_rel_emb_0 = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_0.device)
    expanded_rel_emb_0 = xavier_uniform_norm_(expanded_rel_emb_0)
    expanded_rel_emb_0 = clamp_norm_(expanded_rel_emb_0,maxnorm=1, p=2, dim=-1)

    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb_0[old_idx]
    

    # Relations 1
    old_rel_emb_1 = model.relation_representations[1]._embeddings.weight.data
    expanded_rel_emb_1 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_1.device)
    expanded_rel_emb_1 = xavier_uniform_norm_(expanded_rel_emb_1)
    expanded_rel_emb_1 = clamp_norm_(expanded_rel_emb_1,maxnorm=1, p=2, dim=-1)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_1[new_relation_to_id[r]] = old_rel_emb_1[old_idx]

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'entity_representations.1._embeddings.weight', 'relation_representations.1._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}


    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = TransD(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.entity_representations[1]._embeddings.weight.copy_(expanded_ent_emb_1)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_rel_emb_1)

        
    return new_model


def expand_TorusE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities 0
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    nn.init.xavier_uniform_(expanded_ent_emb_0, gain=nn.init.calculate_gain("relu"))
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]

    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations 0
    old_rel_emb_0 = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_0.device)
    nn.init.xavier_uniform_(expanded_rel_emb_0, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb_0[old_idx]
     

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = TorusE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)

        
    return new_model


def expand_MuRE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities 0
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    nn.init.normal_(expanded_ent_emb_0)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]

    
    # Entities 1
    old_ent_emb_1 = model.entity_representations[1]._embeddings.weight.data
    expanded_ent_emb_1 = torch.empty(len(new_entity_to_id), 1, device=old_ent_emb_1.device)
    nn.init.zeros_(expanded_ent_emb_1)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_1[new_entity_to_id[e]] = old_ent_emb_1[old_idx]
    
    # Entities 2
    old_ent_emb_2 = model.entity_representations[2]._embeddings.weight.data
    expanded_ent_emb_2 = torch.empty(len(new_entity_to_id), 1, device=old_ent_emb_2.device)
    nn.init.zeros_(expanded_ent_emb_2)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_2[new_entity_to_id[e]] = old_ent_emb_2[old_idx]


    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)
    expanded_ent_emb_1 = initialize(initialization,expanded_ent_emb_1,initialization_args,old_entities, new_entities, new_entity_to_id)
    expanded_ent_emb_2 = initialize(initialization,expanded_ent_emb_2,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations 0
    old_rel_emb_1 = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb_1 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_1.device)
    nn.init.normal_(expanded_rel_emb_1)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_1[new_relation_to_id[r]] = old_rel_emb_1[old_idx]

    # Relations 1
    old_rel_emb_2 = model.relation_representations[1]._embeddings.weight.data
    expanded_rel_emb_2 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_2.device)
    nn.init.uniform_(expanded_rel_emb_2)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_2[new_relation_to_id[r]] = old_rel_emb_2[old_idx]
    
    

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'entity_representations.1._embeddings.weight', 'relation_representations.1._embeddings.weight',
                       'entity_representations.2._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = MuRE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.entity_representations[1]._embeddings.weight.copy_(expanded_ent_emb_1)
        new_model.entity_representations[2]._embeddings.weight.copy_(expanded_ent_emb_2)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_1)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_rel_emb_2)

        
    return new_model

def expand_QuatE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(size=(len(new_entity_to_id),model.entity_representations[0].shape[0],model.entity_representations[0].shape[1]), device=old_ent_emb.device)
    expanded_ent_emb = init_quaternions(expanded_ent_emb)
    expanded_ent_emb = expanded_ent_emb.reshape(len(new_entity_to_id), -1)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(size=(len(new_relation_to_id),model.relation_representations[0].shape[0],model.relation_representations[0].shape[1]), device=old_rel_emb.device)
    expanded_rel_emb = init_quaternions(expanded_rel_emb)
    expanded_rel_emb = expanded_rel_emb.reshape(len(new_relation_to_id), -1)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]

    

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}  


    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = QuatE(
        triples_factory=new_tf,
        embedding_dim=model.relation_representations[0].shape[0],
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model

def expand_PairRE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities 0
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    nn.init.uniform_(expanded_ent_emb_0)
    expanded_ent_emb_0 = torch.nn.functional.normalize(expanded_ent_emb_0, p=2, dim=-1)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]

    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)
    
    # Relations 0
    old_rel_emb_0 = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_0.device)
    nn.init.uniform_(expanded_rel_emb_0)

    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb_0[old_idx]
    

    # Relations 1
    old_rel_emb_1 = model.relation_representations[1]._embeddings.weight.data
    expanded_rel_emb_1 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_1.device)
    nn.init.uniform_(expanded_rel_emb_1)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_1[new_relation_to_id[r]] = old_rel_emb_1[old_idx]

        
 
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'relation_representations.1._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = PairRE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_rel_emb_1)

        
    return new_model

def expand_CP(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):

    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities 0
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    nn.init.xavier_uniform_(expanded_ent_emb_0, gain=nn.init.calculate_gain("relu"))
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]

    
    # Entities 1
    old_ent_emb_1 = model.entity_representations[1]._embeddings.weight.data
    expanded_ent_emb_1 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_1.device)
    nn.init.xavier_uniform_(expanded_ent_emb_1, gain=nn.init.calculate_gain("relu"))

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_1[new_entity_to_id[e]] = old_ent_emb_1[old_idx]


    # Relations 0
    old_rel_emb_0 = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_0.device)
    nn.init.xavier_uniform_(expanded_rel_emb_0, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb_0[old_idx]
    

    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)
    expanded_ent_emb_1 = initialize(initialization,expanded_ent_emb_1,initialization_args,old_entities, new_entities, new_entity_to_id)


    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'entity_representations.1._embeddings.weight']
    
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    new_model = CP(
        triples_factory=new_tf,
        embedding_dim=model.entity_representations[0].shape[1],
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.entity_representations[1]._embeddings.weight.copy_(expanded_ent_emb_1)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)

        
    return new_model


def expand_CrossE(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data.clone()
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.xavier_uniform_(expanded_ent_emb, gain=nn.init.calculate_gain("relu"))

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data.clone()
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    nn.init.xavier_uniform_(expanded_rel_emb_0, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb[old_idx]
    

    # Relations
    old_rel_emb = model.relation_representations[1]._embeddings.weight.data.clone()
    expanded_rel_emb_1 = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    nn.init.xavier_uniform_(expanded_rel_emb_1, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_1[new_relation_to_id[r]] = old_rel_emb[old_idx]
    
    
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'relation_representations.1._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = CrossE(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_rel_emb_1)
    
    return new_model


def expand_DistMA(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.xavier_uniform_(expanded_ent_emb, gain=nn.init.calculate_gain("relu"))

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    nn.init.xavier_uniform_(expanded_rel_emb, gain=nn.init.calculate_gain("relu"))
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]

    
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = DistMA(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)
        
    return new_model


def expand_ConvKB(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):

    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    nn.init.uniform_(expanded_ent_emb_0)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]
    

    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(model.relation_representations[0]._embeddings.weight.shape[0], rel_dim, device=old_rel_emb.device)
    nn.init.uniform_(expanded_rel_emb)

    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]



        
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 
                        'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = ConvKB(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
        
    )


    new_model.load_state_dict(filtered_state_dict, strict=False)

    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)

        
    return new_model

def expand_KG2E(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    rel_dim = model.relation_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities 0
    old_ent_emb_0 = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb_0 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_0.device)
    nn.init.uniform_(expanded_ent_emb_0)
    expanded_ent_emb_0 = clamp_norm_(expanded_ent_emb_0,maxnorm=1, p=2, dim=-1)
    

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_0[new_entity_to_id[e]] = old_ent_emb_0[old_idx]

    
    # Entities 1
    old_ent_emb_1 = model.entity_representations[1]._embeddings.weight.data
    expanded_ent_emb_1 = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb_1.device)
    nn.init.uniform_(expanded_ent_emb_1)
    expanded_ent_emb_1 = torch.clamp(expanded_ent_emb_1,min=0.05, max=5)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb_1[new_entity_to_id[e]] = old_ent_emb_1[old_idx]


    expanded_ent_emb_0 = initialize(initialization,expanded_ent_emb_0,initialization_args,old_entities, new_entities, new_entity_to_id)
    expanded_ent_emb_1 = initialize(initialization,expanded_ent_emb_1,initialization_args,old_entities, new_entities, new_entity_to_id)

    # Relations 0
    old_rel_emb_0 = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb_0 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_0.device)
    nn.init.uniform_(expanded_rel_emb_0)
    expanded_rel_emb_0 = clamp_norm_(expanded_rel_emb_0,maxnorm=1, p=2, dim=-1)

    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_0[new_relation_to_id[r]] = old_rel_emb_0[old_idx]
    

    # Relations 1
    old_rel_emb_1 = model.relation_representations[1]._embeddings.weight.data
    expanded_rel_emb_1 = torch.empty(len(new_relation_to_id), rel_dim, device=old_rel_emb_1.device)
    nn.init.uniform_(expanded_rel_emb_1)
    expanded_rel_emb_1 = torch.clamp(expanded_rel_emb_1,min=0.05, max=5)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb_1[new_relation_to_id[r]] = old_rel_emb_1[old_idx]



    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight',
                       'entity_representations.1._embeddings.weight', 'relation_representations.1._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    new_model = KG2E(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )

    
    new_model.load_state_dict(filtered_state_dict, strict=False)
    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb_0)
        new_model.entity_representations[1]._embeddings.weight.copy_(expanded_ent_emb_1)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb_0)
        new_model.relation_representations[1]._embeddings.weight.copy_(expanded_rel_emb_1)

        
    return new_model


def expand_ERMLP(model, new_entity_to_id, entity_to_id, new_relation_to_id, relation_to_id, initialization, initialization_args, train_path):
    # Expand embeddings
    emb_dim = model.entity_representations[0]._embeddings.weight.shape[1]
    old_entities = set(entity_to_id.keys())
    old_relations = set(relation_to_id.keys())
    new_entities = set(new_entity_to_id.keys()) - old_entities
    new_relations = set(new_relation_to_id.keys()) - old_relations

    # Entities
    old_ent_emb = model.entity_representations[0]._embeddings.weight.data
    expanded_ent_emb = torch.empty(len(new_entity_to_id), emb_dim, device=old_ent_emb.device)
    nn.init.uniform_(expanded_ent_emb)

    # Copy old embeddings directly
    for e, old_idx in entity_to_id.items():
        expanded_ent_emb[new_entity_to_id[e]] = old_ent_emb[old_idx]



    # Relations
    old_rel_emb = model.relation_representations[0]._embeddings.weight.data
    expanded_rel_emb = torch.empty(len(new_relation_to_id), emb_dim, device=old_rel_emb.device)
    nn.init.uniform_(expanded_rel_emb)
    
    for r, old_idx in relation_to_id.items():
        expanded_rel_emb[new_relation_to_id[r]] = old_rel_emb[old_idx]

    expanded_ent_emb = initialize(initialization,expanded_ent_emb,initialization_args,old_entities, new_entities, new_entity_to_id)

        
    # Store old state dict
    keys_to_exclude = ['entity_representations.0._embeddings.weight', 'relation_representations.0._embeddings.weight']
    old_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k,v in old_state_dict.items() if k not in keys_to_exclude}

    # Create new TF with expanded mappings
    new_tf = TriplesFactory.from_path(
        train_path,
        entity_to_id=new_entity_to_id,
        relation_to_id=new_relation_to_id
    )

    # Expand model inheriting weights

    new_model = ERMLP(
        triples_factory=new_tf,
        embedding_dim=emb_dim,
        random_seed =  model._random_seed
    )
    
    new_model.load_state_dict(filtered_state_dict, strict=False)
    with torch.no_grad():
        new_model.entity_representations[0]._embeddings.weight.copy_(expanded_ent_emb)
        new_model.relation_representations[0]._embeddings.weight.copy_(expanded_rel_emb)
        
    return new_model