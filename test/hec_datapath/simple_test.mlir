func.func @simple_test(%a: i1, %b: i1) -> i1 {
  // Simple test: double negation elimination
  // NOT(NOT(a)) should equal a
  
  %true = arith.constant true
  %not_a = arith.xori %a, %true : i1
  %not_not_a = arith.xori %not_a, %true : i1
  
  return %not_not_a : i1
} 