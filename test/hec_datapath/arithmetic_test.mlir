func.func @arithmetic_test(%a: i32) -> (i32, i32) {
  // Test arithmetic equivalences from HEC paper Table 1
  // Expression 1: Multiplication by 2
  // Expression 2: Left shift by 1
  // These should be equivalent: a * 2 = a << 1
  
  %c2 = arith.constant 2 : i32
  %c1 = arith.constant 1 : i32
  
  // Expression 1: Multiply by 2
  %mul_result = arith.muli %a, %c2 : i32
  
  // Expression 2: Left shift by 1 (equivalent to multiply by 2)
  %shift_result = arith.shli %a, %c1 : i32
  
  // Return both results - they should be equivalent
  return %mul_result, %shift_result : i32, i32
} 