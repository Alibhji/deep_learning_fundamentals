flowchart TD
    SRC["Source (B, N_src, d)"] --> ENC["Encoder Stack"]
    ENC --> MEM["Encoded Memory (B, N_src, d)"]
    TGT["Target (B, N_tgt, d)"] --> DEC["Decoder Stack"]
    MEM --> DEC
    DEC --> HEAD["Output Projection (vocab)"]
