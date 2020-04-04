/**
 * represent a tensor with specified dimensions and data type.
 */
export interface Tensor {
  /**
   * Get the dimensions of the tensor.
   */
  readonly dims: ReadonlyArray<number>;
  /**
   * Get the data type of the tensor.
   */
  readonly type: Tensor.Type;
  /**
   * Get the number of elements in the tensor.
   */
  readonly size: number;
  /**
   * Get the buffer data of the tensor.
   */
  readonly data: Tensor.DataType;
}

/**
 * represent a tensor with specified dimensions and data type.
 */
export interface TypedTensor<T extends Tensor.Type> extends Tensor {
  /**
   * Get the data type of the tensor.
   */
  readonly type: T;

  /**
   * Get the buffer data of the tensor.
   */
  readonly data: Tensor.DataTypeMap[T]
}

export interface Tensor {
  // Tensor utility functions

  reshape(dims: ReadonlyArray<number>): Tensor;
}

export interface TypedTensor<T extends Tensor.Type> {
  // Tensor utility functions

  reshape(dims: ReadonlyArray<number>): TypedTensor<T>;
}

export declare namespace Tensor {
  interface DataTypeMap {
    float32: Float32Array;
    uint8: Uint8Array;
    int8: Int8Array;
    uint16: Uint16Array;
    int16: Int16Array;
    int32: Int32Array;
    int64: BigInt64Array;
    string: string[];
    bool: Uint8Array;
    float16: never;  // hold on using Uint16Array before we have a concret solution for float 16
    float64: Float64Array;
    uint32: Uint32Array;
    uint64: BigUint64Array;
    // complex64: never;
    // complex128: never;
    // bfloat16: never;
  }

  interface ElementTypeMap {
    float32: number;
    uint8: number;
    int8: number;
    uint16: number;
    int16: number;
    int32: number;
    int64: number;  // may lose presicion
    string: string;
    bool: boolean;
    float16: never;  // hold on before we have a concret solution for float 16
    float64: number;
    uint32: number;
    uint64: number;  // may lose presicion
    // complex64: never;
    // complex128: never;
    // bfloat16: never;
  }

  type DataType = DataTypeMap[Type];
  type ElementType = ElementTypeMap[Type];

  /**
   * represent the data type of a tensor
   */
  export type Type = keyof DataTypeMap;
}


export interface TensorConstructor {
  new<T extends Tensor.Type>(data: Tensor.DataTypeMap[T], type: T, dims?: ReadonlyArray<number>): TypedTensor<T>;
}

// TBD: not implemented yet, use trick to make TypeScript compiler happy
export const Tensor: TensorConstructor = {} as unknown as TensorConstructor;
