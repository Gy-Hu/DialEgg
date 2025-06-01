#!/bin/bash

# HEC Datapath Verification Setup Test Script
# This script verifies that all examples from the README work correctly

echo "🚀 HEC Datapath Verification Test Suite"
echo "========================================"
echo

# Test 1: Boolean Algebra (Exercise 1)
echo "📋 Test 1: Boolean Algebra (De Morgan's Laws)"
echo "Expected: num matches 2 and num matches 3"
echo "Command: egglog test/hec_datapath/manual_extract_test.egg"
echo

echo "Running Boolean test..."
BOOLEAN_OUTPUT=$(egglog test/hec_datapath/manual_extract_test.egg 2>&1)
BOOLEAN_MATCHES=$(echo "$BOOLEAN_OUTPUT" | grep "num matches" | wc -l)
DEMORGAN_MATCHES=$(echo "$BOOLEAN_OUTPUT" | grep "num matches 2" | wc -l)
DOUBLE_NEG_MATCHES=$(echo "$BOOLEAN_OUTPUT" | grep "num matches 3" | wc -l)

if [ $DEMORGAN_MATCHES -ge 1 ] && [ $DOUBLE_NEG_MATCHES -ge 1 ]; then
    echo "✅ Boolean Algebra Test PASSED"
    echo "   - De Morgan's law applied successfully"
    echo "   - Double negation elimination working"
else
    echo "❌ Boolean Algebra Test FAILED"
    echo "   Check manual_extract_test.egg file"
fi
echo

# Test 2: Arithmetic Transformations (Exercise 2)
echo "📋 Test 2: Arithmetic Transformations"
echo "Expected: 24+ total rule matches"
echo "Command: egglog test/hec_datapath/arithmetic_manual_test.egg"
echo

echo "Running arithmetic test..."
ARITH_OUTPUT=$(egglog test/hec_datapath/arithmetic_manual_test.egg 2>&1)
TOTAL_MATCHES=$(echo "$ARITH_OUTPUT" | grep "num matches" | awk '{sum += $NF} END {print sum}')
SHIFT_MUL_MATCHES=$(echo "$ARITH_OUTPUT" | grep -E "num matches [35]" | wc -l)

if [ $TOTAL_MATCHES -ge 20 ] && [ $SHIFT_MUL_MATCHES -ge 2 ]; then
    echo "✅ Arithmetic Transformations Test PASSED"
    echo "   - Total rule matches: $TOTAL_MATCHES"
    echo "   - Shift↔Multiplication equivalence verified"
    echo "   - Associativity rules working"
else
    echo "❌ Arithmetic Transformations Test FAILED"
    echo "   Total matches: $TOTAL_MATCHES (expected 20+)"
    echo "   Check arithmetic_manual_test.egg file"
fi
echo

# Test 3: File Structure Check
echo "📋 Test 3: File Structure Verification"
echo "Checking all required files exist..."

REQUIRED_FILES=(
    "README.md"
    "manual_extract_test.egg"
    "arithmetic_manual_test.egg"
    "nand_baseline.mlir"
    "demorgan_equivalent.mlir"
    "arithmetic_test.mlir"
    "boolean_rules.egg"
    "arithmetic_rules.egg"
)

MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "test/hec_datapath/$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (MISSING)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -eq 0 ]; then
    echo "✅ File Structure Test PASSED"
else
    echo "❌ File Structure Test FAILED - $MISSING_FILES files missing"
fi
echo

# Summary
echo "🎯 SUMMARY"
echo "=========="

TOTAL_TESTS=3
PASSED_TESTS=0

if [ $DEMORGAN_MATCHES -ge 1 ] && [ $DOUBLE_NEG_MATCHES -ge 1 ]; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

if [ $TOTAL_MATCHES -ge 20 ] && [ $SHIFT_MUL_MATCHES -ge 2 ]; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

if [ $MISSING_FILES -eq 0 ]; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

echo "Tests Passed: $PASSED_TESTS/$TOTAL_TESTS"
echo

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo "Your HEC datapath verification implementation is working correctly."
    echo "You can now:"
    echo "  - Follow the tutorial in README.md"
    echo "  - Run individual exercises from the documentation"
    echo "  - Experiment with your own transformation rules"
    echo
    echo "📚 Next steps: Read README.md for detailed explanations and advanced exercises."
else
    echo "⚠️  Some tests failed. Please check:"
    echo "  - DialEgg is properly installed and built"
    echo "  - egglog is in your PATH"
    echo "  - All required files are present"
    echo "  - Run the failing tests manually for more details"
fi
echo

echo "🔧 Quick Commands:"
echo "  Boolean test:    egglog test/hec_datapath/manual_extract_test.egg"
echo "  Arithmetic test: egglog test/hec_datapath/arithmetic_manual_test.egg"
echo "  Read tutorial:   cat test/hec_datapath/README.md"
echo 