import {InferenceSession,
    OnnxValue,
    Tensor,
    TypedTensor} from '../lib';

//
// constructors
//
{
    // default option
    const mySessionA = new InferenceSession();
    // empty option
    const mySessionB = new InferenceSession({});
    // option object
    const myOptionC: InferenceSession.SessionOptions = { interOpNumThreads: 4 };
    const mySessionC = new InferenceSession(myOptionC);

    // NOTE: more tests about session option is in inference-session-options.ts
}

//
// Load model
//
{
    (async () => {
        const mySession: InferenceSession = new InferenceSession();
        try {
            // NOTE: following codes are demonstration of overrides. should not call loadModel() multiple times

            // load from file
            await mySession.loadModel('C:\\models\\test.onnx');

            // OR load from an array buffer
            const someBuffer = new ArrayBuffer(100000); // suppose its content is filled
            await mySession.loadModel(someBuffer);

            // OR a part of the buffer
            await mySession.loadModel(someBuffer, 400, 6400);

            // OR a Uint8Array
            const someUint8Array = new Uint8Array(1000);
            await mySession.loadModel(someUint8Array);
        }
        catch (e) {
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

        const mySession: InferenceSession = new InferenceSession();
        await mySession.loadModel('C:\\my_model.onnx');

        // NOTE: please DO use try-catch with await. this code is a demonstration purpose for method override only so we want to make it shorter.

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

        const input0: Tensor = new Tensor(bufferData0, 'float32', [2, 5, 10]);
        const input1: Tensor = new Tensor(bufferData1, 'float32', [2, 3, 5]);

        //
        // overrides for input feed
        //

        // following code will demonstrate how to feed input to a run() call

        // pass `Feed` interface
        const result00 = await mySession.run({
            names: ['data0', 'data1'],
            values: [input0, input1]
        });

        // pass values only. `names` can be omitted, if it's the same as model's input name list, and you are very sure about it.
        const result01 = await mySession.run({
            values: [input0, input1]
        });

        // an easier way to pass values only.
        const result02 = await mySession.run([input0, input1]);

        // besides putting names and values separated, we can use an object
        const result03 = await mySession.run({
            'data0': input0,
            'data1': input1
        });

        // use a ES6 Map object
        const feedMap = new Map<string, OnnxValue>();
        feedMap.set('data0', input0);
        feedMap.set('data1', input1);
        const result04 = await mySession.run(feedMap);

        // all the above calls can work with optional RunOption parameter:
        const result05 = await mySession.run(feedMap, { logSeverityLevel: 1 });


        //
        // overrides for output fetches
        //

        // besides overrides for input feed, those can be combined with several output fetches.
        // output fetches are optional. when omitted, the inference engine will output all outputs
        // in the model's output definition, and allocate buffers if necessary for each output tensors.
        // if you want to use a pre-allocated buffer as output, you need to pass the output via fetches.

        const inputFeed = {}; // input is whatever type that run() accepts.

        // specify output names
        const result10 = await mySession.run(inputFeed, ['output0', 'output1']);

        // specify output names via `Fetch` interface
        const result11 = await mySession.run(inputFeed, {
            names: ['output0', 'output1']
        });

        // if want output0 only:
        const result12 = await mySession.run(inputFeed, ['output0']);


        // pre-allocated buffer
        //
        const bufferOutput0 = new Int32Array(5);
        const output0: Tensor = new Tensor(bufferOutput0, 'int32', [1, 5]);

        // pass pre-allocated value via `Fetch` interface
        const result13 = await mySession.run(inputFeed, {
            names: ['output0', 'output1'],
            values: [output0, null] // null is necessary. it indicates that do not use pre-allocated value for 'output1'
        });

        // via Map
        const outputFetch = new Map<string, InferenceSession.NullableOnnxValue>();
        outputFetch.set('output0', output0);
        outputFetch.set('output1', null);
        const result14 = await mySession.run(inputFeed, outputFetch);

        //
        // using result
        //

        // the result is a map object.
        const resultOutput0: OnnxValue = result14.get('output0');
        const resultOutput1: OnnxValue = result14.get('output1');

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
        const mySession: InferenceSession = new InferenceSession();
        await mySession.loadModel('C:\\my_model.onnx');

        // get input/output names
        const inputNames = mySession.inputNames;
        const outputNames = mySession.outputNames;
    })();
}