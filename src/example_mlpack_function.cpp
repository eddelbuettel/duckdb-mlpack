
#include "example_mlpack_function.hpp"

namespace duckdb {

// Example scalar function accessing mlpack header
void MlpackMlpackVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "Mlpack " + name.GetString() + ", my included mlpack version is " + mlpack::util::GetVersion() );
        });
}

}
