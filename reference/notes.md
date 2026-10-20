# Notes

Put the paper PDF (2509.26507v1.pdf) in this folder.

## Eq. 6 (one layer, one letter)
    x     = x + relu((Gx_e - Gx_i) @ y)        Y pushes X      (round 4l+3)
    a     = sigma @ x                           read memory     (round 4l)
    sigma = u * (sigma + outer(y, x) * Gs)      write memory    (round 4l+1)
    y     = relu((Gy_e - Gy_i) @ a) * x         A pushes Y      (round 4l+2)

Trained: Gx_e, Gx_i, Gy_e, Gy_i (all >= 0), embed, head. Gs fixed at first.
Not trained: sigma (memory, starts at 0 for every text).

## Two kinds of facts
- Always true ("banana makes you happy"): learned in the G's.
- True only in this story ("tom eats a banana"): stored in sigma.
