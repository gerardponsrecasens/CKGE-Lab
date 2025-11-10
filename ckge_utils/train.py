import os
import json
import torch
import pickle
import pandas as pd
import gradio as gr
import plotly.express as px
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.models import *
from pykeen.training import SLCWATrainingLoop
from pykeen.optimizers import Adam
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from .expand_models import *
from .helper import *

def build_plot_for_test_snapshot(test_snapshot, metric, total_results, metric_dict):

    n_train = len(total_results)
    n_test = len(total_results)
    test_snapshot = int(test_snapshot)
    if test_snapshot < 0 or test_snapshot >= n_test:
        fig = px.bar(title="Invalid test snapshot")
        return fig

    labels = []
    values = []
    for train_snapshot in range(test_snapshot, n_train):
        try:
            val = total_results[train_snapshot][test_snapshot]['tail']['realistic'].get(metric, None)
        except Exception:
            val = None
        if val is not None:
            labels.append(f"S{train_snapshot}")
            values.append(val)

    if len(values) == 0:
        fig = px.bar(title=f"No data for metric '{metric}' at test snapshot {test_snapshot}")
        return fig

    # ---- One Color + Cleaner Look ----
    fig = px.bar(
        x=labels,
        y=values,
        labels={'x': 'Training Snapshot', 'y': metric_dict[metric]},
        title=f"Test Snapshot {test_snapshot}"
    )

    fig.update_traces(
        hovertemplate="Train: %{x}<br>Value: %{y:.4f}",
        marker_color="#4C72B0",   # <-- Single color (blue toned)
        marker_line_width=0.5,
        marker_line_color="black"
    )

    fig.update_layout(
        template="simple_white",
        title_font=dict(size=22, family="Arial", color="#333"),
        title_x=0.5,                # <-- Center the title
        font=dict(size=14),
        xaxis_tickangle=-30,
        yaxis=dict(gridcolor="rgba(180,180,180,0.3)")
    )

    return fig


def train_CKGE(embedding_model,dataset_name,number_snapshots,embedding_dimension,lr_0,lr_inc,epochs_0, epochs_inc, random_seed,initialization_args={'method':'random'}, inverse_triples = False, dashboarding = False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random_noise = 0
    initialization = initialization_args['method']
    if initialization != 'random':
        init_args = initialization_args['args']
        if 'random_noise' in init_args.keys():
            random_noise = init_args['random_noise']
    else:
        init_args = {}
                  
    # Train base model on snapshot 0
    snapshot = 0
    snapshot_dir = f"data/{dataset_name}/{snapshot}"

    total_results = {}
    test_list = []
    train_list = []
    valid_list = []

    # Load triples
    train_tf, valid_tf, test_tf = create_triples_factory(dataset_name, snapshot, inverse_triples=inverse_triples)
    test_list.append(test_tf)
    train_list.append(train_tf)
    valid_list.append(valid_tf)

    model_classes = {
        'transe': TransE,
        'transh': TransH,
        'rotate': RotatE,
        'transr': TransR,
        'distmult': DistMult,
        'hole': HolE,
        'proje': ProjE,
        'complex': ComplEx,
        'rescal': RESCAL,
        'transf': TransF,
        'tucker': TuckER,
        'conve': ConvE,
        'simple': SimplE,
        'boxe': BoxE,
        'transd': TransD,
        'toruse': TorusE,
        'pairre': PairRE,
        'cp': CP,
        'mure': MuRE,
        'quate': QuatE,
        'crosse': CrossE,
        'distma': DistMA,
        'convkb': ConvKB,
        'kg2e': KG2E,
        'ermlp': ERMLP}
    
    key = embedding_model.lower()
    if key not in model_classes:
        raise ValueError(f"Not supported embedding model: {embedding_model}")

    model = model_classes[key](
        triples_factory=train_tf,
        embedding_dim=embedding_dimension,
        random_seed=random_seed,
    )


    
    model.to(device)

    # Train with early stopping
    optimizer = Adam(params=model.get_grad_params(), lr=lr_0)
    stopper = EarlyStopper(
        model=model,
        evaluator=RankBasedEvaluator(),
        training_triples_factory=train_tf,
        evaluation_triples_factory=valid_tf,
        patience=2,
        frequency=10,
        metric="hits@3"
    )
    trainer = SLCWATrainingLoop(model=model, optimizer=optimizer, triples_factory=train_tf)
    trainer.train(triples_factory=train_tf, num_epochs=epochs_0, batch_size=256, stopper=stopper)

    # Evaluate on test triples
    evaluator = RankBasedEvaluator()
    results = evaluator.evaluate(model, mapped_triples=test_tf.mapped_triples, additional_filter_triples=[train_tf.mapped_triples, valid_tf.mapped_triples])

    total_results[snapshot] = [results.to_dict()]

    # Save everything for next snapshot
    save_snapshot(model, train_tf, snapshot_dir)


    # Step 2: Expand and train on snapshot 1

    for snapshot in range(1,number_snapshots):
        model, entity_to_id, relation_to_id = switch_snapshot(
            model=model,
            entity_to_id=train_tf.entity_to_id,
            relation_to_id=train_tf.relation_to_id,
            model_name=embedding_model,
            snapshot=snapshot,
            dataset_name=dataset_name,
            initialization=initialization,
            initialization_args=init_args,
            inverse_triples=inverse_triples
        )
        
        model.to(device)

        train_tf, valid_tf, test_tf = create_triples_factory(dataset_name, snapshot, entity_to_id, relation_to_id, inverse_triples=inverse_triples)
        test_list.append(test_tf)
        train_list.append(train_tf)
        valid_list.append(valid_tf)

        # Train again with early stopping
        optimizer = Adam(params=model.get_grad_params(),lr=lr_inc)
        stopper = EarlyStopper(
            model=model,
            evaluator=RankBasedEvaluator(),
            training_triples_factory=train_tf,
            evaluation_triples_factory=valid_tf,
            patience=2,
            frequency=10,
            metric="hits@3",
        )
        trainer = CKGETrainingLoop(model=model, optimizer=optimizer, triples_factory=train_tf)
        trainer.train(triples_factory=train_tf, num_epochs=epochs_inc, batch_size=256, stopper=stopper)



        # Evaluate
        partial_results = []
        filter_train = torch.cat([i.mapped_triples for i in train_list], dim=0)
        filter_valid = torch.cat([i.mapped_triples for i in valid_list], dim=0)

        for j, t_tf in enumerate(test_list):
            evaluator = RankBasedEvaluator()
            results = evaluator.evaluate(model, mapped_triples=t_tf.mapped_triples, additional_filter_triples=[filter_train, filter_valid])
            partial_results.append(results.to_dict())

        total_results[snapshot] = partial_results
            
        # Save model + updated mappings
        snapshot_dir = f"data/{dataset_name}/{snapshot}"
        save_snapshot(model, train_tf, snapshot_dir)

    

    metrics = get_metrics(number_snapshots, test_list, total_results)

    # Store the results

    report = {'dataset': dataset_name, 'emb_dim': embedding_dimension, 'seed': random_seed, 'init': initialization,
            'RN': random_noise, 'model': embedding_model, 'lr_0': lr_0, 'lr_inc': lr_inc, 'total_results': total_results, 'metrics':metrics}

    report_path = './results/' + dataset_name + '_' + embedding_model + '_' + initialization + '_' +str(lr_inc)+'_'+ str(random_noise)+ '_' +str(random_seed)

    with open(report_path+'.json', 'w') as f:
        json.dump(report, f)

    
    if dashboarding:
        metric_names = ['hits_at_1', 'hits_at_3', 'hits_at_10', 'inverse_harmonic_mean_rank']
        metric_dict = {'hits_at_1':'Hits@1', 'hits_at_3':'Hits@3', 'hits_at_10':'Hits@10', 'inverse_harmonic_mean_rank':'MRR'}



        order = ["mrr", "hits@1", "hits@3", "hits@10", "cf_mrr", "new_mrr"]
        rename_map = {
            "mrr": "MRR",
            "hits@1": "Hits@1",
            "hits@3": "Hits@3",
            "hits@10": "Hits@10",
            "cf_mrr": "Ω_old",
            "new_mrr": "Ω_new"
        }
        n_test = len(total_results)
        metrics_df = pd.DataFrame([[metrics[k] for k in order]], columns=[rename_map[k] for k in order]).round(3)

        

        initial_fig = build_plot_for_test_snapshot(0, metric_names[0], total_results, metric_dict)

        with gr.Blocks(css="""
            body {background-color: #f7f9fb;}
            .gradio-container {max-width: 1000px;}
        """) as demo:

            gr.Markdown("<h1 style='text-align: center;'>📊 CKGE Dashboard</h1>")

            total_results_state = gr.State(total_results)
            metric_dict_state = gr.State(metric_dict)
            
            with gr.Row():
                gr.Dataframe(value=metrics_df, interactive=False, wrap=True, label="Metrics Summary")

            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=3):
                    test_slider = gr.Slider(
                        minimum=0,
                        maximum=max(0, n_test - 1),
                        step=1,
                        value=0,
                        label="Test snapshot"
                    )
                    metric_dropdown = gr.Dropdown(
                        label="Metric",
                        choices=metric_names,
                        value=metric_names[1]
                    )
                    gr.Markdown("Select the test snapshot `s`.")
                with gr.Column(scale=7):
                    plot_out = gr.Plot(value=initial_fig)

            # callbacks
            test_slider.change(fn=build_plot_for_test_snapshot, inputs=[test_slider, metric_dropdown, total_results_state, metric_dict_state], outputs=plot_out)
            metric_dropdown.change(fn=build_plot_for_test_snapshot, inputs=[test_slider, metric_dropdown, total_results_state, metric_dict_state], outputs=plot_out)

        demo.launch(debug=True)
    
    
    metrics = pd.DataFrame({k: [v] for k, v in metrics.items()}).round(3)
    cols_to_keep = ['cf_mrr', 'new_mrr','mrr', 'hits@1', 'hits@3', 'hits@10']
    metrics = metrics[cols_to_keep]
    metrics.rename(columns={'cf_mrr': 'Ω_old', 'new_mrr': 'Ω_new'}, inplace=True)


    
    return model, metrics