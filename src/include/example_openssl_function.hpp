#pragma once

#include "duckdb.hpp"

#include <openssl/opensslv.h>			// OpenSSL linked through vcpkg

namespace duckdb {

	void MlpackOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result);

}
