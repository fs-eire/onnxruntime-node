import {OnnxValue} from './onnx-value';

/**
 * Represent a runtime instance of an ONNX model.
 */
export interface InferenceSession {
  //#region run()

  /**
   * Execute the model asynchronously with the given feeds and options.
   * @param feeds Representation of the model input. See type description of `InferenceSession.InputType` for detail.
   * @param options Optional. A set of options that controls the behavior of model inference.
   */
  run(feeds: InferenceSession.InputType,
      options?: InferenceSession.RunOptions): Promise<InferenceSession.OnnxValueMapType>;

  /**
   * Execute the model asynchronously with the given feeds, fetches and options.
   * @param feeds Representation of the model input. See type description of `InferenceSession.InputType` for detail.
   * @param fetches Representation of the model output. See type description of `InferenceSession.OutputType` for
   *     detail.
   * @param options Optional. A set of options that controls the behavior of model inference.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding values.
   */
  run(feeds: InferenceSession.InputType, fetches: InferenceSession.OutputType,
      options?: InferenceSession.RunOptions): Promise<InferenceSession.OnnxValueMapType>;

  //#endregion run()

  //#region metadata

  /**
   * Get input names of the loaded model.
   */
  readonly inputNames: ReadonlyArray<string>;

  /**
   * Get output names of the loaded model.
   */
  readonly outputNames: ReadonlyArray<string>;

  // /**
  //  * Get input metadata of the loaded model.
  //  */
  // readonly inputMetadata: ReadonlyArray<Readonly<InferenceSession.ValueMetadata>>;

  // /**
  //  * Get output metadata of the loaded model.
  //  */
  // readonly outputMetadata: ReadonlyArray<Readonly<InferenceSession.ValueMetadata>>;

  //#endregion metadata
}

export declare namespace InferenceSession {
  //#region input/output types

  type OnnxValueMapType = {readonly [name: string]: OnnxValue};
  type NullableOnnxValueMapType = {readonly [name: string]: OnnxValue | null};

  /**
   * A model input is an object that use input names as keys and OnnxValue as corresponding values.
   */
  type InputType = OnnxValueMapType;

  /**
   * A model output could be one of the following:
   *
   * - Omitted.
   *     - Use model's output names definition.
   * - An array of string indicating the output names.
   * - An object that use output names as keys and OnnxValue or null as corresponding values.
   *
   * REMARK: different from input argument, in output, OnnxValue is optional. If an OnnxValue is present it will be
   *         used as a pre-allocated value by the inference engine; if omitted, inference engine will allocate buffer
   *         internally.
   */
  type OutputType = ReadonlyArray<string>|NullableOnnxValueMapType;

  type OutputReturnType = {readonly [name: string]: OnnxValue};

  //#endregion input/output types

  //#region session options

  /**
   * A set of configurations for session behavior.
   */
  export interface SessionOptions {
    /**
     * An array of execution provider options.
     *
     * An execution provider option can be a string indicating the name of the execution provider,
     * or an object of corresponding type.
     */
    executionProviders?: ReadonlyArray<SessionOptions.ExecutionProviderConfig>;

    /**
     * The intra OP threads number.
     */
    intraOpNumThreads?: number;

    /**
     * The inter OP threads number.
     */
    interOpNumThreads?: number;

    /**
     * The optimization level.
     */
    graphOptimizationLevel?: 'disabled'|'basic'|'extended'|'all';

    /**
     * Whether enable CPU memory arena.
     */
    enableCpuMemArena?: boolean;

    /**
     * Whether enable memory pattern.
     */
    enableMemPattern?: boolean;

    /**
     * Execution mode.
     */
    executionMode?: 'sequential'|'parallel';

    /**
     * Log ID.
     */
    logId?: string;

    /**
     * Log severity level. See
     * https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/common/logging/severity.h
     */
    logSeverityLevel?: 0|1|2|3|4;
  }

  export namespace SessionOptions {
    //#region execution providers

    // currently we only have CPU, DNNL and CUDA EP. in future to support more.
    interface ExecutionProviderOptionMap {
      cpu: CpuExecutionProviderOption, cuda: CudaExecutionProviderOption
    }

    type ExecutionProviderName = keyof ExecutionProviderOptionMap;
    type ExecutionProviderConfig = ExecutionProviderOptionMap[ExecutionProviderName]|ExecutionProviderName;

    export interface CpuExecutionProviderOption {
      readonly name: 'cpu';
      useArena?: boolean;
    }
    export interface CudaExecutionProviderOption {
      readonly name: 'cuda';
      deviceId?: number;
    }
    //#endregion execution providers
  }

  //#endregion session options

  //#region run options

  /**
   * A set of configurations for inference run behavior
   */
  export interface RunOptions {
    /**
     * Log severity level. See
     * https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/common/logging/severity.h
     */
    logSeverityLevel?: 0|1|2|3|4;

    /**
     * A tag for the Run() calls using this
     */
    tag?: string;
  }

  //#endregion run options

  //#region value metadata

  interface ValueMetadata {
    // TBD
  }

  //#endregion value metadata
}

export interface InferenceSessionConstructor {
  //#region create()

  /**
   * Create a new inference session and load model asynchronously from an ONNX model file.
   * @param path The file path of the model to load.
   * @param options specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(path: string, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from an array bufer.
   * @param buffer An ArrayBuffer representation of an ONNX model.
   * @param options specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: ArrayBufferLike, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from segment of an array bufer.
   * @param buffer An ArrayBuffer representation of an ONNX model.
   * @param byteOffset The beginning of the specified portion of the array buffer.
   * @param length The length in bytes of the array buffer.
   * @param options specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: ArrayBufferLike, byteOffset: number, length?: number, options?: InferenceSession.SessionOptions):
      Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from a Uint8Array.
   * @param buffer A Uint8Array representation of an ONNX model.
   * @param options specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  //#endregion create()
}

// TBD: not implemented yet, use trick to make TypeScript compiler happy
export const InferenceSession: InferenceSessionConstructor = {} as unknown as InferenceSessionConstructor;
