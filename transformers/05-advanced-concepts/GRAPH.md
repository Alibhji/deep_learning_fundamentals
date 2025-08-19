flowchart TD
    X["Tokens (B, N, d)"] --> SELECT["Choose Attention Variant"]
    SELECT --> STD["Standard (O(N^2))"]
    SELECT --> LIN["Linear (O(N))"]
    SELECT --> SPR["Sparse (O(N))"]
    SELECT --> LOC["Local (O(N·w))"]
    STD --> OUT(("Output"))
    LIN --> OUT
    SPR --> OUT
    LOC --> OUT
