
#include "example_scalar_function.hpp"

namespace duckdb {

// Example scalar function
void MlpackScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "Mlpack " + name.GetString() + " 🐥");;
        });
}

}
