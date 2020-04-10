import {InferenceSession, OnnxValue, Tensor, TypedTensor} from '../lib';

//
// Load model
//
{
  (async () => {
    try {
      // load from file
      const mySession0: InferenceSession = await InferenceSession.create('C:\\my_model.onnx');

      // OR load from an array buffer
      const someBuffer = new ArrayBuffer(100000);  // suppose its content is filled
      const mySession1: InferenceSession = await InferenceSession.create(someBuffer);

      // OR a part of the buffer
      const mySession2: InferenceSession = await InferenceSession.create(someBuffer, 400, 6400);

      // OR a Uint8Array
      const someUint8Array = new Uint8Array(1000);
      const mySession3: InferenceSession = await InferenceSession.create(someUint8Array);

      // any of the overrides above accept session option:

      // default option
      const mySession10 = await InferenceSession.create('C:\\my_model.onnx');
      // empty option
      const mySession11 = await InferenceSession.create('C:\\my_model.onnx', {});
      // option object
      const myOption12: InferenceSession.SessionOptions = {interOpNumThreads: 4, logSeverityLevel: 0};
      const mySession12 = await InferenceSession.create('C:\\my_model.onnx', myOption12);

      // NOTE: more tests about session option is in inference-session-options.ts

    } catch (e) {
      // error handling
    }
  })();
}

//
// run
//
{
  (async () => {
    // first we demonstrate how to pass input. refer to tensor.ts for how to construct a tensor.

    const mySession = await InferenceSession.create('C:\\my_model.onnx');

    // NOTE: please DO use try-catch with await. this code is a demonstration purpose for method override only so we
    // want to make it shorter.

    // suppose the model's input and output is like:
    //          | name     | type      | dim
    //   Input
    //        0 | data0    | float32   | [2,5,10]
    //        1 | data1    | float32   | [2,3,5]
    //   Output
    //        0 | output0  | int32     | [1,5]
    //        1 | output1  | float32   | [5]

    // preparing inputs
    const bufferData0 = new Float32Array(100);
    const bufferData1 = new Float32Array(30);

    const input0: Tensor = new Tensor(bufferData0, [2, 5, 10]);
    const input1: Tensor = new Tensor(bufferData1, [2, 3, 5]);

    //
    // overrides for input feed
    //

    // following code will demonstrate how to feed input to a run() call

    // use an object to pass input
    const result03 = await mySession.run({'data0': input0, 'data1': input1});

    // can also work with optional RunOption parameter:
    const result05 = await mySession.run({'data0': input0, 'data1': input1}, {logSeverityLevel: 1});


    //
    // overrides for output fetches
    //

    // besides overrides for input feed, those can be combined with several output fetches.
    // output fetches are optional. when omitted, the inference engine will output all outputs
    // in the model's output definition, and allocate buffers if necessary for each output tensors.
    // if you want to use a pre-allocated buffer as output, you need to pass the output via fetches.

    const inputFeed = {};  // input is whatever type that run() accepts.

    // specify output names only
    const result10 = await mySession.run(inputFeed, ['output0', 'output1']);

    // if want output0 only:
    const result11 = await mySession.run(inputFeed, ['output0']);

    // this is not allowed, output names cannot be an empty array, if not omitted:
    const result12 = await mySession.run(inputFeed, []);  // runtime error


    // pre-allocated buffer
    //
    const bufferOutput0 = new Int32Array(5);
    const output0: Tensor = new Tensor(bufferOutput0, [1, 5]);

    // pass pre-allocated value
    const result13 = await mySession.run(inputFeed, {
      output0: output0,
      output1: null  // null is necessary. it indicates that do not use pre-allocated value for 'output1'
    });

    //
    // using result
    //

    // the result is a map object.
    const resultOutput0: OnnxValue = result13.output0;
    const resultOutput1: OnnxValue = result13.output1;

    // we know that output1 is a float tensor, so
    const myOutput1_A = resultOutput1 as TypedTensor<'float32'>;
    const myData1_A: Float32Array = myOutput1_A.data;

    // using TypedTensor is not necessary. We can use it in an alternative way. 2 ways are the same
    const myOutput1_B = resultOutput1;
    const myData1_B: Float32Array = myOutput1_B.data as Float32Array;

    // now we get the Float32Array so it's ready to use.
  })();
}


//
// metadata
//
{
  (async () => {
    const mySession: InferenceSession = await InferenceSession.create('C:\\my_model.onnx');

    // get input/output names
    const inputNames = mySession.inputNames;    // ['input0', 'input1']
    const outputNames = mySession.outputNames;  // ['output0', 'output1']
  })();
}