# HEC Datapath Verification Examples

Examples from "HEC: Equivalence Verification Checking for Code Transformation via Equality Saturation".

## Setup

```bash
export PATH=$PATH:$(pwd)/egglog/target/release
```

## Example 1: Basic De Morgan's Law (fig1_listing1_2)

Verifies: `NAND(a,b) ≡ OR(NOT(a), NOT(b))`

```bash
./build/egg-opt --eq-sat test/hec_datapath/fig1_listing1_2/test_both.mlir \
  --egg test/hec_datapath/fig1_listing1_2/check_eclass.egg
```

Success: `[INFO] Checked fact ... (= op4 op7)`

## Example 2: With Memory Operations (loop_with_datapath)

```bash
./build/egg-opt --eq-sat test/hec_datapath/loop_with_datapath/simple_loop_test.mlir \
  --egg test/hec_datapath/loop_with_datapath/loop_rules.egg
```

Success: `[INFO] Checked fact ... (= op6 op9)`

## Example 3: XOR Expansion (xor_expansion)

Verifies: `a ⊕ b ≡ (a ∧ ¬b) ∨ (¬a ∧ b)`

```bash
./build/egg-opt --eq-sat test/hec_datapath/xor_expansion/xor_simple.mlir \
  --egg test/hec_datapath/xor_expansion/xor_working_rules.egg
```

Success: `[INFO] Checked fact ... (= op3 op8)`

This example demonstrates XOR can be expanded into basic AND, OR, and NOT operations.

## Directory Structure

- `fig1_listing1_2/` - Basic NAND and De Morgan's law equivalence verification
- `loop_with_datapath/` - More complex example with memory operations
- `xor_expansion/` - XOR operation expansion into basic logical operations