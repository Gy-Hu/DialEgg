func.func @nand_test(%av: memref<101xi1>, %bv: memref<101xi1>) {
  // Baseline NAND implementation - corresponds to paper's Listing 1
  // Performs NAND(a, b) operation on two vectors of length 101
  
  affine.for %arg1 = 0 to 101 {
    // Load values from input vectors
    %a = affine.load %av[%arg1] : memref<101xi1>
    %b = affine.load %bv[%arg1] : memref<101xi1>
    
    // Compute AND operation
    %and_result = arith.andi %a, %b : i1
    
    // Compute NAND by negating AND result (XOR with true)
    %true = arith.constant true
    %nand_result = arith.xori %and_result, %true : i1
    
    // Store result back to first vector
    affine.store %nand_result, %av[%arg1] : memref<101xi1>
  }
  
  return
} 