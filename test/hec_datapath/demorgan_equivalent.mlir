func.func @demorgan_test(%av: memref<101xi1>, %bv: memref<101xi1>) {
  // De Morgan's law equivalent implementation - corresponds to paper's Listing 3
  // Transforms NAND(a, b) to OR(NOT(a), NOT(b))
  // Should be functionally equivalent to nand_baseline.mlir
  
  affine.for %arg1 = 0 to 101 {
    // Load values from input vectors
    %a = affine.load %av[%arg1] : memref<101xi1>
    %b = affine.load %bv[%arg1] : memref<101xi1>
    
    // Apply De Morgan's law: NAND(a, b) = OR(NOT(a), NOT(b))
    %true = arith.constant true
    
    // Compute NOT(a) and NOT(b)
    %not_a = arith.xori %a, %true : i1
    %not_b = arith.xori %b, %true : i1
    
    // Compute OR(NOT(a), NOT(b))
    %or_result = arith.ori %not_a, %not_b : i1
    
    // Store result back to first vector
    affine.store %or_result, %av[%arg1] : memref<101xi1>
  }
  
  return
} 