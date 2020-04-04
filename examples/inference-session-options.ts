import {InferenceSession} from '../lib';

// every property in SessionOptions is optional. so we can have an empty object:
const option0: InferenceSession.SessionOptions = {};

//
// execution provider
//

// one execution provider option can be a string or an object with more details
const option1_0: InferenceSession.SessionOptions = {
  executionProviders: ['cpu']
};
const option1_1: InferenceSession.SessionOptions = {
  executionProviders: ['cuda', 'cpu']
};
const option1_2: InferenceSession.SessionOptions = {
  executionProviders: [{name: 'cuda', deviceId: 0}, {name: 'cpu'}]
};
const option1_3: InferenceSession.SessionOptions = {
  executionProviders: [{name: 'cuda', deviceId: 0}, 'cpu']
};

// various of flags and settings
const option3: InferenceSession.SessionOptions = {
  enableCpuMemArena: true,
  enableMemPattern: true,
  intraOpNumThreads: 4,
  interOpNumThreads: 2
};

// graph optimization level
const option4: InferenceSession.SessionOptions = {
  graphOptimizationLevel: 'extended'
};

// execution mode
const option5: InferenceSession.SessionOptions = {
  executionMode: 'sequential'
};
