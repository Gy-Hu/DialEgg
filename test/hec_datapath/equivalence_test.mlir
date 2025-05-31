func.func @equivalence_test(%a: i1, %b: i1) -> (i1, i1) {
  // Two equivalent expressions that should be unified by De Morgan's law:
  // Expression 1: NAND(a, b) = XOR(AND(a, b), true)
  // Expression 2: OR(NOT(a), NOT(b)) = OR(XOR(a, true), XOR(b, true))
  
  // Expression 1: NAND implementation
  %and_result = arith.andi %a, %b : i1
  %true = arith.constant true
  %nand_result = arith.xori %and_result, %true : i1
  
  // Expression 2: De Morgan's law equivalent
  %not_a = arith.xori %a, %true : i1
  %not_b = arith.xori %b, %true : i1
  %demorgan_result = arith.ori %not_a, %not_b : i1
  
  // Return both results - they should be equivalent
  return %nand_result, %demorgan_result : i1, i1
} 