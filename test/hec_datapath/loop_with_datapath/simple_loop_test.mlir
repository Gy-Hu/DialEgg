// Simple loop test: verify NAND and De Morgan equivalence for a single iteration
func.func @simple_loop_test(%av: memref<1xi1>, %bv: memref<1xi1>) -> (i1, i1) {
  // Load values from memory
  %a = affine.load %av[0] : memref<1xi1>
  %b = affine.load %bv[0] : memref<1xi1>
  
  // Expression 1: NAND(a,b)
  %and = arith.andi %a, %b : i1
  %true = arith.constant true
  %nand = arith.xori %and, %true : i1
  
  // Expression 2: OR(NOT(a), NOT(b))
  %not_a = arith.xori %a, %true : i1
  %not_b = arith.xori %b, %true : i1
  %or_result = arith.ori %not_a, %not_b : i1
  
  return %nand, %or_result : i1, i1
} 