func.func @loop_hoisting_test(%av: memref<101xi1>, %bv: memref<101xi1>) {
  // Loop hoisting variant - corresponds to paper's Figure 1 Listing 2
  // The variable %true is moved inside the loop body (hoisted into loop)
  // Should be functionally equivalent to nand_baseline.mlir
  
  affine.for %arg1 = 0 to 101 {
    // Load values from input vectors
    %a = affine.load %av[%arg1] : memref<101xi1>
    %b = affine.load %bv[%arg1] : memref<101xi1>
    
    // Compute AND operation
    %and_result = arith.andi %a, %b : i1
    
    // Variable %true is now defined inside the loop (hoisted)
    %true = arith.constant true
    %nand_result = arith.xori %and_result, %true : i1
    
    // Store result back to first vector
    affine.store %nand_result, %av[%arg1] : memref<101xi1>
  }
  
  return
} 