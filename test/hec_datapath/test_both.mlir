// Test both expressions in one function to see if they're in the same e-class
func.func @test_equivalence(%a: i1, %b: i1) -> (i1, i1) {
  // Expression 1: NAND(a,b)
  %and1 = arith.andi %a, %b : i1
  %true = arith.constant true
  %nand = arith.xori %and1, %true : i1
  
  // Expression 2: OR(NOT(a), NOT(b))
  %not_a = arith.xori %a, %true : i1
  %not_b = arith.xori %b, %true : i1
  %or_result = arith.ori %not_a, %not_b : i1
  
  return %nand, %or_result : i1, i1
} 