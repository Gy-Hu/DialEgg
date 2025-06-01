#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @_2mm(%arg0: tensor<100x10xi64>, %arg1: tensor<10x150xi64>, %arg2: tensor<150x8xi64>) -> tensor<100x8xi64> {
    %0 = tensor.empty() : tensor<100x150xi64>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<100x10xi64>, tensor<10x150xi64>) outs(%0 : tensor<100x150xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %4 = arith.muli %in, %in_0 : i64
      %5 = arith.addi %out, %4 : i64
      linalg.yield %5 : i64
    } -> tensor<100x150xi64>
    %2 = tensor.empty() : tensor<100x8xi64>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %arg2 : tensor<100x150xi64>, tensor<150x8xi64>) outs(%2 : tensor<100x8xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %4 = arith.muli %in, %in_0 : i64
      %5 = arith.addi %out, %4 : i64
      linalg.yield %5 : i64
    } -> tensor<100x8xi64>
    return %3 : tensor<100x8xi64>
  }
}

