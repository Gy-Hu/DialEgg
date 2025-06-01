func.func @xor_test(%a: i1, %b: i1) -> (i1, i1) {
  %true = arith.constant true
  
  // Version 1: Direct XOR
  %xor_direct = arith.xori %a, %b : i1
  
  // Version 2: XOR expanded as (a AND NOT b) OR (NOT a AND b)
  %not_a = arith.xori %a, %true : i1
  %not_b = arith.xori %b, %true : i1
  %a_and_not_b = arith.andi %a, %not_b : i1
  %not_a_and_b = arith.andi %not_a, %b : i1
  %xor_expanded = arith.ori %a_and_not_b, %not_a_and_b : i1
  
  return %xor_direct, %xor_expanded : i1, i1
} 