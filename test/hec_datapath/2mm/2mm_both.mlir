func.func @_2mm_both(%x: tensor<100x10xi64>, %y: tensor<10x150xi64>, %z: tensor<150x8xi64>) -> (tensor<100x8xi64>, tensor<100x8xi64>) {
    // Expression 1: (x * y) * z
    // Cost: 100 * 150 * (10 + 8) = 2,700,000
    %xy_init = tensor.empty() : tensor<100x150xi64>
    %xy = linalg.matmul ins(%x, %y : tensor<100x10xi64>, tensor<10x150xi64>) 
                        outs(%xy_init : tensor<100x150xi64>) -> tensor<100x150xi64>
    
    %xy_z_init = tensor.empty() : tensor<100x8xi64>
    %xy_z = linalg.matmul ins(%xy, %z : tensor<100x150xi64>, tensor<150x8xi64>) 
                          outs(%xy_z_init : tensor<100x8xi64>) -> tensor<100x8xi64>
    
    // Expression 2: x * (y * z)
    // Cost: 10 * 8 * (150 + 100) = 20,000
    %yz_init = tensor.empty() : tensor<10x8xi64>
    %yz = linalg.matmul ins(%y, %z : tensor<10x150xi64>, tensor<150x8xi64>)
                        outs(%yz_init : tensor<10x8xi64>) -> tensor<10x8xi64>
    
    %x_yz_init = tensor.empty() : tensor<100x8xi64>
    %x_yz = linalg.matmul ins(%x, %yz : tensor<100x10xi64>, tensor<10x8xi64>)
                          outs(%x_yz_init : tensor<100x8xi64>) -> tensor<100x8xi64>
    
    return %xy_z, %x_yz : tensor<100x8xi64>, tensor<100x8xi64>
} 