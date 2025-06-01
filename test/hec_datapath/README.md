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

## Quick Tips

- View operations: `grep "FROM OP:" output`
- Check `.ops.egg` for operation numbers
- Use `(check (= opX opY))` to verify equivalence

## Directory Structure

- `fig1_listing1_2/` - Basic NAND and De Morgan's law equivalence verification
- `loop_with_datapath/` - More complex example with memory operations

## Understanding the Output

1. **Operation List**: The output shows the numbering for each operation:
   ```
   [6] (arith_xori op4 op5 (I1)) FROM OP: %3 = arith.xori %2, %true : i1
   ```

2. **Rule Matching**: Shows how many times the De Morgan's law rule matched:
   ```
   num matches 1
   ```

3. **Equivalence Check**: Most importantly, the result of the `check` command confirms that two operations are in the same e-class.

## File Descriptions

- `*.mlir` - MLIR input files containing code to verify
- `*.egg` - E-graph rewriting rule files
- `*.ops.egg` - Auto-generated intermediate files (containing concrete operations)
- `*-egglog.log` - Detailed execution logs

## Creating Your Own Examples

To create your own verification examples:

1. Write an MLIR function containing two equivalent expressions
2. Create a corresponding `.egg` rule file
3. Use `(check (= opX opY))` to verify equivalence

## Troubleshooting

- If you see "egglog: not found", ensure PATH is set correctly
- Check `.ops.egg` files to understand operation numbering
- Use `grep "FROM OP:"` to quickly view operation mappings 