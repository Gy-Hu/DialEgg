func.func @simple_unrolling_original(%arr: memref<4xi32>) {
  // Original loop: simple iteration over 4 elements
  // This should be equivalent to the unrolled version
  
  affine.for %i = 0 to 4 {
    %val = affine.load %arr[%i] : memref<4xi32>
    %c1 = arith.constant 1 : i32
    %incremented = arith.addi %val, %c1 : i32
    affine.store %incremented, %arr[%i] : memref<4xi32>
  }
  
  return
}

func.func @simple_unrolling_unrolled(%arr: memref<4xi32>) {
  // Unrolled version: manually unrolled loop
  // Should be equivalent to the original loop
  
  // Iteration 0
  %val0 = affine.load %arr[0] : memref<4xi32>
  %c1_0 = arith.constant 1 : i32
  %inc0 = arith.addi %val0, %c1_0 : i32
  affine.store %inc0, %arr[0] : memref<4xi32>
  
  // Iteration 1
  %val1 = affine.load %arr[1] : memref<4xi32>
  %c1_1 = arith.constant 1 : i32
  %inc1 = arith.addi %val1, %c1_1 : i32
  affine.store %inc1, %arr[1] : memref<4xi32>
  
  // Iteration 2  
  %val2 = affine.load %arr[2] : memref<4xi32>
  %c1_2 = arith.constant 1 : i32
  %inc2 = arith.addi %val2, %c1_2 : i32
  affine.store %inc2, %arr[2] : memref<4xi32>
  
  // Iteration 3
  %val3 = affine.load %arr[3] : memref<4xi32>
  %c1_3 = arith.constant 1 : i32
  %inc3 = arith.addi %val3, %c1_3 : i32
  affine.store %inc3, %arr[3] : memref<4xi32>
  
  return
} 