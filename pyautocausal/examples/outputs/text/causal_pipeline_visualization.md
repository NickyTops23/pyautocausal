# Graph Visualization

## Executable Graph

```mermaid
graph TD
    node0[df]
    node1{multi_period}
    node2{single_treated_unit}
    node3{multi_post_periods}
    node4{stag_treat}
    node5[stand_spec]
    node6[ols_stand]
    node7[ols_stand_output]
    node8[synthdid_spec]
    node9[synthdid_fit]
    node10[synthdid_plot]
    node11[hainmueller_fit]
    node12[hainmueller_placebo]
    node13[hainmueller_output]
    node14[did_spec]
    node15[ols_did]
    node16[save_ols_did]
    node17[save_did_output]
    node18[event_spec]
    node19[ols_event]
    node20[event_plot]
    node21[save_event_output]
    node22[stag_spec]
    node23[ols_stag]
    node24{has_never_treated}
    node25[cs_never_treated]
    node26[cs_never_treated_event_plot]
    node27[cs_never_treated_group_plot]
    node28[cs_not_yet_treated]
    node29[cs_not_yet_treated_event_plot]
    node30[cs_not_yet_treated_diagnostics]
    node31[stag_event_plot]
    node32[save_stag_output]
    node0 --> node1
    node1 -->|True| node2
    node1 -->|False| node5
    node1 -->|True| node14
    node2 -->|False| node3
    node2 -->|True| node8
    node3 -->|True| node4
    node4 -->|False| node18
    node4 -->|True| node22
    node5 --> node6
    node6 --> node7
    node8 --> node9
    node8 --> node11
    node9 --> node10
    node11 --> node12
    node12 --> node13
    node14 --> node15
    node15 --> node16
    node15 --> node17
    node18 --> node19
    node19 --> node20
    node19 --> node21
    node22 --> node23
    node22 --> node24
    node23 --> node31
    node23 --> node32
    node24 -->|True| node25
    node24 -->|False| node28
    node25 --> node26
    node25 --> node27
    node28 --> node29
    node28 --> node30

    %% Node styling
    classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;
    classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;
    classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;
    classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;
    classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;
    style node0 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node1 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node2 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node3 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node4 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node5 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node6 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node7 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node8 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node9 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node10 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node11 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node12 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node13 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node14 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node15 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node16 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node17 fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black
    style node18 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node19 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node20 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node21 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node22 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node23 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node24 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node25 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node26 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node27 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node28 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node29 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node30 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node31 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
    style node32 fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black
```

## Node Legend

### Node Types
```mermaid
graph LR
    actionNode[Action Node] ~~~ decisionNode{Decision Node}
    style actionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black
    style decisionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black
```

### Node States
```mermaid
graph LR
    pendingNode[Pending]:::pendingNode ~~~ runningNode[Running]:::runningNode ~~~ completedNode[Completed]:::completedNode ~~~ failedNode[Failed]:::failedNode ~~~ passedNode[Passed]:::passedNode

    classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;
    classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;
    classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;
    classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;
    classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;
```

Node state coloring indicates the execution status of each node in the graph.
