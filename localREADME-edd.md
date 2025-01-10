
### Step One

after initial (i.e. unmodified) build and test of binary and unittests,
ran 'python3 ./scripts/bootstrap-template.py mlpack'
re-ran 'make'
validated new binary:

```sh
edd@rob:~/git/duckdb-extension-template(main)$ build/release/duckdb
v1.1.3 19864453f7
Enter ".help" for usage hints.
Connected to a transient in-memory database.
Use ".open FILENAME" to reopen on a persistent database.
D select mlpack('well hello') as result;
┌──────────────────────┐
│        result        │
│       varchar        │
├──────────────────────┤
│ Mlpack well hello 🐥 │
└──────────────────────┘
D
```

Also added `ccache` to top-level CMakeLists.txt 'just in case'


### Step Two

Added CMake/Findmlpack.cmake, activated it in CMakeLists.txt
Added minimal src/mlpack_extension.cpp along with addition to vcpkg.json

Then

```sh
edd@rob:~/git/duckdb-extension-template(main)$ build/release/duckdb
v1.1.3 19864453f7
Enter ".help" for usage hints.
Connected to a transient in-memory database.
Use ".open FILENAME" to reopen on a persistent database.
D select mlpack_mlpack_version('ryan and dirk');
┌──────────────────────────────────────────────────────────────────┐
│              mlpack_mlpack_version('ryan and dirk')              │
│                             varchar                              │
├──────────────────────────────────────────────────────────────────┤
│ Mlpack ryan and dirk, my included mlpack version is mlpack 4.5.1 │
└──────────────────────────────────────────────────────────────────┘
D
```

### Step Three

Expanded the (still just one) key source file to return a (static, fixed,
boring) table

edd@rob:~/git/duckdb-extension-mlpack(main)$ echo "select * from mlpack_table();" |  build/release/duckdb
┌───────┬───────┬───────┬────────┐
│ col_1 │ col_2 │ col_3 │ col_4  │
│ int16 │ int32 │ int64 │ double │
├───────┼───────┼───────┼────────┤
│     0 │     1 │     2 │    3.0 │
│     4 │     5 │     6 │    7.0 │
│     8 │     9 │    10 │   11.0 │
└───────┴───────┴───────┴────────┘
edd@rob:~/git/duckdb-extension-mlpack(main)$ 
