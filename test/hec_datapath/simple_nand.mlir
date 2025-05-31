func.func @simple_nand(%a: i1, %b: i1) -> i1 {
  // Simple NAND operation for testing Boolean algebra rules
  // This should be equivalent to OR(NOT(a), NOT(b)) via De Morgan's law
  
  // Compute AND operation
  %and_result = arith.andi %a, %b : i1
  
  // Compute NAND by negating AND result (XOR with true)
  %true = arith.constant true
  %nand_result = arith.xori %and_result, %true : i1
  
  return %nand_result : i1
} 