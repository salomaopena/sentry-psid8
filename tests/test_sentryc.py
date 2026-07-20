#!/usr/bin/env python3
"""SENTRY-C tests (pure numpy/networkx; no GPU, no torch required).

The central test is the HARD CONSTRAINT: accident, fire and crime may never seed
a correlation subgraph, but they MUST be reachable as consequences of a valid
seed. Everything else in this file exists to keep that guarantee honest.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentryc import (NetworkEvent, ListEventSource, EventGraphBuilder, Asset,
                     FusionJustification, SEED_ELIGIBLE, CONSEQUENCE_ONLY,
                     SeedConstraintError, CLASS_ID, TemporalRuleScorer,
                     build_alert)
from sentryc.graph_builder import EDGE_PRECEDES, NODE_NETWORK
from sentryc.metrics import (lead_time_gain, false_positive_reduction,
                             modality_contribution, convergent_event_prf)


def _alert(cls, t0, t1, conf=0.8, asset="cam_01"):
    return {"class_id": CLASS_ID[cls], "t_start": t0, "t_end": t1,
            "confidence": conf, "asset_id": asset, "track": [],
            "confidence_trajectory": [conf], "evidence_terms": {"motion_gate_mean": 0.5}}


def _assets():
    return [Asset("cam_01", zone="lobby", segment="vlan10"),
            Asset("cam_02", zone="lobby", segment="vlan10")]


def test_class_partition_is_total_and_disjoint():
    assert SEED_ELIGIBLE.isdisjoint(CONSEQUENCE_ONLY)
    assert len(SEED_ELIGIBLE) == 5 and len(CONSEQUENCE_ONLY) == 3
    assert SEED_ELIGIBLE == {"intrusion", "suspicious_behavior", "vandalism",
                             "fall", "suspicious_object"}
    assert CONSEQUENCE_ONLY == {"accident", "fire", "crime"}
    print("test_class_partition_is_total_and_disjoint OK")


def test_seeding_with_primary_classes_succeeds():
    b = EventGraphBuilder(delta_t=300, assets=_assets())
    for cls in sorted(SEED_ELIGIBLE):
        phys = [_alert(cls, 100, 140)]
        G = b.build(phys, [], seeds=[0])
        assert G.nodes["P0"]["seed"] is True, cls
    print("test_seeding_with_primary_classes_succeeds OK  (5 classes)")


def test_seeding_with_consequence_class_raises():
    b = EventGraphBuilder(delta_t=300, assets=_assets())
    for cls in sorted(CONSEQUENCE_ONLY):
        phys = [_alert(cls, 100, 140)]
        try:
            b.build(phys, [], seeds=[0])
        except SeedConstraintError as e:
            assert cls in str(e)
        else:
            raise AssertionError(f"seeding with {cls!r} must raise SeedConstraintError")
    print("test_seeding_with_consequence_class_raises OK  (accident, crime, fire)")


def test_default_seeds_exclude_consequence_classes():
    """Auto-seeding must silently skip consequence-only events, not crash and not
    include them."""
    b = EventGraphBuilder(delta_t=300, assets=_assets())
    phys = [_alert("fire", 100, 140), _alert("intrusion", 100, 140)]
    G = b.build(phys, [])                       # seeds=None -> auto
    assert G.nodes["P0"]["seed"] is False       # fire: never a seed
    assert G.nodes["P1"]["seed"] is True        # intrusion: seed
    print("test_default_seeds_exclude_consequence_classes OK")


def test_consequence_class_reachable_as_successor():
    """The other half of the constraint: fire/crime/accident MUST be able to
    appear downstream of a valid seed through a `precedes` edge."""
    b = EventGraphBuilder(delta_t=600, assets=_assets())
    phys = [_alert("vandalism", 100, 140),      # seed
            _alert("fire", 200, 260)]           # consequence, strictly after
    G = b.build(phys, [])
    assert G.nodes["P0"]["seed"] is True and G.nodes["P1"]["seed"] is False
    edges = [e for _, _, e in G.edges(data=True) if e.get("etype") == EDGE_PRECEDES]
    assert any(e["dt"] > 0 for e in edges), "seed must precede the consequence"
    subs = b.correlation_subgraphs(G)
    assert "P1" in subs["P0"].nodes, "fire must be reachable from the vandalism seed"
    print("test_consequence_class_reachable_as_successor OK")


def test_precedes_requires_positive_delta():
    """A consequence that happened BEFORE the seed is not a consequence."""
    b = EventGraphBuilder(delta_t=600, assets=_assets())
    phys = [_alert("vandalism", 300, 340), _alert("fire", 100, 160)]
    G = b.build(phys, [])
    subs = b.correlation_subgraphs(G)
    assert "P1" not in subs["P0"].nodes, "a prior fire must not attach to a later seed"
    print("test_precedes_requires_positive_delta OK")


def test_network_edges_and_window():
    b = EventGraphBuilder(delta_t=100, assets=_assets())
    phys = [_alert("intrusion", 500, 540)]
    net = [NetworkEvent(450, "cam_01", "port_scan", 0.9),      # 50s before -> in
           NetworkEvent(900, "cam_01", "exfiltration", 0.8)]   # 360s after -> out
    G = b.build(phys, net)
    linked = {nb for _, nb, e in G.out_edges("P0", data=True)
              if G.nodes[nb].get("ntype") == NODE_NETWORK}
    assert linked == {"N0"}, f"only the in-window event may link, got {linked}"
    print("test_network_edges_and_window OK")


def test_fusion_justification_filters_pairs():
    """gain = P(e_n|e_p,class) - P(e_n); a pair below theta must not be linked."""
    just = FusionJustification(
        p_joint={("intrusion", "port_scan"): 0.60, ("intrusion", "new_device"): 0.05},
        p_base={"port_scan": 0.05, "new_device": 0.04},
        amb_physical={"intrusion": 0.9},
        amb_network={"port_scan": 0.5, "new_device": 0.5})
    assert abs(just.gain("intrusion", "port_scan") - 0.55) < 1e-9
    assert abs(just.gain("intrusion", "new_device") - 0.01) < 1e-9
    b = EventGraphBuilder(delta_t=300, assets=_assets(), justification=just,
                          theta=0.10, tau=0.2)
    phys = [_alert("intrusion", 500, 540)]
    net = [NetworkEvent(480, "cam_01", "port_scan", 0.9),
           NetworkEvent(490, "cam_01", "new_device", 0.9)]
    G = b.build(phys, net)
    linked = {G.nodes[nb]["anomaly_type"] for _, nb, e in G.out_edges("P0", data=True)
              if G.nodes[nb].get("ntype") == NODE_NETWORK}
    assert linked == {"port_scan"}, f"only the informative pair survives theta, got {linked}"
    print("test_fusion_justification_filters_pairs OK")


def test_rule_scorer_and_alert_lead_time():
    b = EventGraphBuilder(delta_t=600, assets=_assets())
    phys = [_alert("intrusion", 500, 540, conf=0.7)]
    net = [NetworkEvent(200, "cam_01", "port_scan", 0.9)]     # reconnaissance first
    G = b.build(phys, net)
    sub = b.correlation_subgraphs(G)["P0"]
    scorer = TemporalRuleScorer(window=600)
    p, sev = scorer.score(sub, "P0")
    assert 0.0 <= p <= 1.0 and 0.0 <= sev <= 1.0
    assert p > 0.7, "network evidence must raise the probability above video alone"
    a = build_alert(sub, "P0", p, sev)
    # first evidence = the port scan (t=200); emission = end of the physical event (540)
    assert a.first_evidence_at == 200 and a.emitted_at == 540
    assert a.time_to_detection == 340
    assert a.correlated_asset == "cam_01"
    assert len(a.network_evidence) == 1 and len(a.physical_evidence) == 1
    print("test_rule_scorer_and_alert_lead_time OK")


def test_scorer_refuses_non_eligible_seed():
    """Defence in depth: even if a graph is assembled elsewhere, the scorer
    re-checks the constraint."""
    b = EventGraphBuilder(delta_t=600, assets=_assets())
    G = b.build([_alert("fire", 100, 140)], [])          # no seeds
    G.nodes["P0"]["seed"] = True                          # forced, bypassing the builder
    try:
        TemporalRuleScorer().score(G, "P0")
    except SeedConstraintError:
        print("test_scorer_refuses_non_eligible_seed OK")
    else:
        raise AssertionError("the scorer must re-check seed eligibility")


def test_convergent_metrics():
    b = EventGraphBuilder(delta_t=600, assets=_assets())
    scorer = TemporalRuleScorer(window=600)
    cid = dict(CLASS_ID)

    # one real incident: intrusion at [500, 540], preceded by a port scan
    phys = [_alert("intrusion", 500, 540, conf=0.7)]
    net = [NetworkEvent(200, "cam_01", "port_scan", 0.9)]
    G = b.build(phys, net)
    sub = b.correlation_subgraphs(G)["P0"]
    p, s = scorer.score(sub, "P0")
    conv = [build_alert(sub, "P0", p, s)]

    # the physical-only detector sees the same event but no network context
    G0 = b.build(phys, [])
    sub0 = b.correlation_subgraphs(G0)["P0"]
    p0, s0 = scorer.score(sub0, "P0")
    uni = [build_alert(sub0, "P0", p0, s0)]

    gts = [{"class_id": cid["intrusion"], "t_start": 500, "t_end": 540}]

    prf = convergent_event_prf(conv, gts, cid, tiou_thr=0.3)
    assert prf[cid["intrusion"]]["tp"] == 1

    lt = lead_time_gain(conv, uni, gts, cid, tiou_thr=0.3)
    assert lt["n_paired"] == 1
    # convergent sees evidence from t=200; physical-only only from t=500
    assert lt["convergent_mean"] > lt["unimodal_mean"], \
        "the convergent chain starts earlier, so its evidence window is longer"

    mc = modality_contribution(conv, gts, cid, tiou_thr=0.3)
    assert mc["joint"] == 1 and mc["joint_fraction"] == 1.0

    fpr = false_positive_reduction([a.as_event(cid[a.cls]) for a in conv],
                                   [a.as_event(cid[a.cls]) for a in uni],
                                   gts, tiou_thr=0.3)
    assert fpr["fp_convergent"] == 0 and fpr["fp_unimodal"] == 0
    print("test_convergent_metrics OK")


def test_network_event_schema_is_closed():
    try:
        NetworkEvent(1.0, "cam_01", "definitely_not_a_type", 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("anomaly_type vocabulary must be closed")
    try:
        NetworkEvent(1.0, "cam_01", "port_scan", 1.7)
    except ValueError:
        pass
    else:
        raise AssertionError("confidence must be validated")
    src = ListEventSource([NetworkEvent(10, "cam_01", "port_scan", 0.5),
                           NetworkEvent(400, "cam_02", "new_device", 0.6)])
    assert len(src.events(0, 100)) == 1 and src.assets() == {"cam_01", "cam_02"}
    print("test_network_event_schema_is_closed OK")


def test_feature_encoding_is_torch_free_and_correct():
    """The graph->feature contract (where GNN bugs hide) is numpy-only, so it is
    verified without a GPU."""
    import numpy as np
    from sentryc import (gather_neighbors, encode_physical, encode_network,
                         PHYS_DIM, NET_DIM)
    from sentryc.graph_builder import SEED_ELIGIBLE

    b = EventGraphBuilder(delta_t=600, assets=_assets())
    phys = [_alert("intrusion", 500, 540, conf=0.7),   # seed
            _alert("fire", 600, 660, conf=0.6)]        # consequence via `precedes`
    net = [NetworkEvent(200, "cam_01", "port_scan", 0.9),          # precedes seed
           NetworkEvent(520, "cam_01", "unauthorized_access", 0.8)] # during seed
    G = b.build(phys, net)
    sub = b.correlation_subgraphs(G)["P0"]

    x_seed, x_neigh, kinds, rels = gather_neighbors(sub, "P0")
    assert x_seed.shape == (PHYS_DIM,)
    assert x_seed[0] == np.float32(0.7), "seed confidence"
    assert x_seed[1] == np.float32(40.0), "seed duration = t_end - t_start"
    assert x_seed[2] == 1.0, "is_seed flag"
    order = sorted(SEED_ELIGIBLE)
    assert x_seed[4 + order.index("intrusion")] == 1.0, "class one-hot"

    # Messages are per EDGE, not per node (documented in gather_neighbors):
    # both network events are on the seed's own asset, so each sends TWO
    # messages - one along temporal_proximity, one along spatial_colocation.
    assert x_neigh.shape[0] == len(kinds) == len(rels)
    assert kinds.count("NetworkEvent") == 4, "2 network events x 2 relations"
    assert kinds.count("PhysicalEvent") == 1, "the fire consequence must arrive"
    assert rels.count("temporal_proximity") == 2
    assert rels.count("spatial_colocation") == 2
    assert rels.count("precedes") == 1, "the consequence arrives via `precedes`"

    # a network event BEFORE the seed must encode a NEGATIVE signed delta:
    # that sign is what lets the model learn "reconnaissance precedes intrusion"
    ni = [i for i, k in enumerate(kinds) if k == "NetworkEvent"]
    deltas = sorted(set(float(x_neigh[i][1]) for i in ni))
    assert deltas == [-300.0, 20.0], (
        f"port scan t=200 -> -300 (precedes); unauthorized_access t=520 -> +20; got {deltas}")
    print("test_feature_encoding_is_torch_free_and_correct OK")


def test_encoding_gives_no_class_onehot_to_consequences():
    """A consequence-only class must not occupy a seed-class one-hot slot; it is
    context, never a correlation target."""
    import numpy as np
    from sentryc import encode_physical, PHYS_DIM
    d = {"cls": "fire", "class_id": 3, "t_start": 10, "t_end": 20,
         "confidence": 0.9, "evidence": {}, "seed": False}
    v = encode_physical(d, 0.0)
    assert v.shape == (PHYS_DIM,)
    assert v[4:].sum() == 0.0, "consequence classes carry no seed-class one-hot"
    print("test_encoding_gives_no_class_onehot_to_consequences OK")


if __name__ == "__main__":
    test_class_partition_is_total_and_disjoint()
    test_seeding_with_primary_classes_succeeds()
    test_seeding_with_consequence_class_raises()
    test_default_seeds_exclude_consequence_classes()
    test_consequence_class_reachable_as_successor()
    test_precedes_requires_positive_delta()
    test_network_edges_and_window()
    test_fusion_justification_filters_pairs()
    test_rule_scorer_and_alert_lead_time()
    test_scorer_refuses_non_eligible_seed()
    test_convergent_metrics()
    test_network_event_schema_is_closed()
    test_feature_encoding_is_torch_free_and_correct()
    test_encoding_gives_no_class_onehot_to_consequences()
    print("\nALL SENTRY-C TESTS PASSED")
