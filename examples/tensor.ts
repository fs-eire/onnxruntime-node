import {Tensor} from '../lib';

//
// constructors
//

// constructors - good usage
{
  // create a [2x3x4] float tensor
  const bufferA = new Float32Array(24);
  bufferA[0] = 0.1;
  const tensorA = new Tensor('float32', bufferA, [2, 3, 4]);
  // type 'float32' can be omitted and the type is inferred from data
  const tensorA_2 = new Tensor(bufferA, [2, 3, 4]);

  // create a [1x2] boolean tensor
  const bufferB = new Uint8Array(2);
  bufferB[0] = 1;  // true
  bufferB[1] = 0;  // false
  const tensorB = new Tensor('bool', bufferB, [1, 2]);

  // create a scaler float64 tensor
  const tensorC = new Tensor(new Float64Array(1), []);

  // create a one-dimension tensor
  const tensorD = new Tensor(new Float32Array(100));  // tensorD.dims = [100]

  // create a [1x2] string tensor
  const tensorE = new Tensor('string', ['a', 'b'], [1, 2]);
  const tensorE_2 = new Tensor(['a', 'b'], [1, 2]);
}
// constructors - bad usage
{
  // ! updated: create from normal array is allowed in new API proposal now
  // // create from normal array
  // const tensorA = new Tensor('int32', [1, 2, 3], [1, 3]);  // BUILD ERROR
  //                                                          // not allowed; all numeric tensors must create from
  //                                                          // TypedArray may support in future by Tensor.createFrom()

  // create from mismatched TypedArray
  const bufferB = new Float32Array(100);
  const tensorB = new Tensor('float64', bufferB);  // BUILD ERROR
                                                   // 'float64' must use with Float64Array as data.

  // bad dimension (non-number)
  const tensorC = new Tensor(new Float32Array(100), [1, 2, 'hello-world']);  // BUILD ERROR
  // bad dimension (non-integer)
  const tensorD = new Tensor(new Float32Array(100), [1, 2, 0.5]);  // BUILD: OK; RUNTIME: should throw
  // bad dimension (negative value)
  const tensorE = new Tensor(new Float32Array(100), [1, 2, -1]);  // BUILD: OK; RUNTIME: should throw

  // size mismatch (scalar size should be 1)
  const tensorF = new Tensor(new Float32Array(0), []);  // should throw
  // size mismatch (5 * 6 != 40)
  const tensorG = new Tensor(new Float32Array(40), [5, 6]);  // should throw
}

//
// utilities
//
{
  // reshape
  const tensorA = new Tensor(new Float32Array(100)).reshape([10, 10]);

  // TBD: add more
}
