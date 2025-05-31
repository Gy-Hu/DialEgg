func.func @simple_demorgan(%a: i1, %b: i1) -> i1 {
  // De Morgan's law equivalent: NAND(a, b) = OR(NOT(a), NOT(b))
  // This should be proven equivalent to simple_nand.mlir by our rules
  
  %true = arith.constant true
  
  // Compute NOT(a) and NOT(b)
  %not_a = arith.xori %a, %true : i1
  %not_b = arith.xori %b, %true : i1
  
  // Compute OR(NOT(a), NOT(b))
  %result = arith.ori %not_a, %not_b : i1
  
  return %result : i1
} 