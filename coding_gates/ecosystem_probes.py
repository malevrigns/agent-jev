"""Hand-written outputs from test tools the benchmark never ran, plus agents reporting in prose.

Fixed before the second fine-tune was scored; read once, at the end, with the build_dataset gate
wording. Every case is unambiguous under that wording's criteria: True only when at least one
test ran and every test that ran passed.
"""

PROBES = [
    ("go_pass", True, "go test ./...",
     "ok  \tgithub.com/acme/billing/invoice\t0.012s\nok  \tgithub.com/acme/billing/tax\t0.009s\n"
     "?   \tgithub.com/acme/billing/cmd\t[no test files]"),
    ("go_fail", False, "go test ./...",
     "--- FAIL: TestApplyDiscount (0.00s)\n    discount_test.go:18: ApplyDiscount(1000, 15) = 860, want 850\n"
     "FAIL\nFAIL\tgithub.com/acme/billing/invoice\t0.011s\nok  \tgithub.com/acme/billing/tax\t0.008s\nFAIL"),
    ("go_build_failed", False, "go test ./...",
     "# github.com/acme/billing/invoice [github.com/acme/billing/invoice.test]\n"
     "invoice/discount.go:14:9: undefined: roundCents\nFAIL\tgithub.com/acme/billing/invoice [build failed]\nFAIL"),
    ("go_no_tests", False, "go test ./...", "?   \tgithub.com/acme/tools\t[no test files]"),
    ("cargo_pass", True, "cargo test",
     "   Compiling slugger v0.3.1 (/home/dev/slugger)\n    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.84s\n"
     "     Running unittests src/lib.rs (target/debug/deps/slugger-5f1c2a9e)\n\nrunning 4 tests\n"
     "test tests::basic ... ok\ntest tests::empty_input ... ok\ntest tests::trailing_dash ... ok\ntest tests::unicode ... ok\n\n"
     "test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"),
    ("cargo_fail", False, "cargo test",
     "running 4 tests\ntest tests::basic ... ok\ntest tests::trailing_dash ... FAILED\ntest tests::unicode ... ok\n"
     "test tests::empty_input ... ok\n\nfailures:\n\n---- tests::trailing_dash stdout ----\n\n"
     "thread 'tests::trailing_dash' panicked at src/lib.rs:41:9:\nassertion `left == right` failed\n"
     "  left: \"hello-world-\"\n right: \"hello-world\"\n\nfailures:\n    tests::trailing_dash\n\n"
     "test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n\n"
     "error: test failed, to rerun pass `--lib`"),
    ("maven_pass", True, "mvn test",
     "[INFO] -------------------------------------------------------\n[INFO]  T E S T S\n"
     "[INFO] -------------------------------------------------------\n[INFO] Running com.acme.auth.TokenServiceTest\n"
     "[INFO] Tests run: 14, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.212 s -- in com.acme.auth.TokenServiceTest\n"
     "[INFO] \n[INFO] Results:\n[INFO] \n[INFO] Tests run: 14, Failures: 0, Errors: 0, Skipped: 0\n[INFO] \n"
     "[INFO] ------------------------------------------------------------------------\n[INFO] BUILD SUCCESS"),
    ("maven_error", False, "mvn test",
     "[ERROR] Tests run: 14, Failures: 0, Errors: 1, Skipped: 0, Time elapsed: 0.231 s <<< FAILURE! -- in com.acme.auth.TokenServiceTest\n"
     "[ERROR] com.acme.auth.TokenServiceTest.verifyExpiredToken -- Time elapsed: 0.012 s <<< ERROR!\n"
     "java.lang.NullPointerException: Cannot invoke \"java.time.Instant.isBefore(java.time.Instant)\" because \"expiry\" is null\n"
     "\tat com.acme.auth.TokenService.verify(TokenService.java:42)\n[INFO] \n[INFO] Results:\n[INFO] \n[ERROR] Errors: \n"
     "[ERROR]   TokenServiceTest.verifyExpiredToken:31 » NullPointer\n[INFO] \n"
     "[ERROR] Tests run: 14, Failures: 0, Errors: 1, Skipped: 0\n[INFO] \n[INFO] BUILD FAILURE"),
    ("gradle_fail", False, "./gradlew test",
     "> Task :app:test FAILED\n\nAuthServiceTest > rejectsExpiredToken() FAILED\n"
     "    org.opentest4j.AssertionFailedError at AuthServiceTest.java:44\n\n12 tests completed, 1 failed\n\n"
     "FAILURE: Build failed with an exception.\n\n* What went wrong:\nExecution failed for task ':app:test'.\n"
     "> There were failing tests. See the report at: file:///home/dev/app/build/reports/tests/test/index.html\n\n"
     "BUILD FAILED in 3s"),
    ("rspec_pass", True, "bundle exec rspec",
     "..........\n\nFinished in 0.04213 seconds (files took 0.3112 seconds to load)\n10 examples, 0 failures"),
    ("rspec_fail", False, "bundle exec rspec",
     "..F.......\n\nFailures:\n\n  1) Cart#total applies the coupon\n     Failure/Error: expect(cart.total).to eq(90)\n\n"
     "       expected: 90\n            got: 100\n\n       (compared using ==)\n"
     "     # ./spec/cart_spec.rb:22:in `block (3 levels) in <top (required)>'\n\n"
     "Finished in 0.0516 seconds (files took 0.30 seconds to load)\n10 examples, 1 failure\n\n"
     "Failed examples:\n\nrspec ./spec/cart_spec.rb:20 # Cart#total applies the coupon"),
    ("rspec_none", False, "bundle exec rspec",
     "No examples found.\n\nFinished in 0.00021 seconds (files took 0.1 seconds to load)\n0 examples, 0 failures"),
    ("phpunit_pass", True, "vendor/bin/phpunit",
     "PHPUnit 11.4.3 by Sebastian Bergmann and contributors.\n\nRuntime:       PHP 8.3.12\n\n"
     "............                                                      12 / 12 (100%)\n\n"
     "Time: 00:00.041, Memory: 8.00 MB\n\nOK (12 tests, 31 assertions)"),
    ("phpunit_fail", False, "vendor/bin/phpunit",
     "............F                                                     13 / 13 (100%)\n\n"
     "Time: 00:00.044, Memory: 8.00 MB\n\nThere was 1 failure:\n\n1) Tests\\PriceTest::testRoundsHalfUp\n"
     "Failed asserting that 10.0 matches expected 11.0.\n\n/app/tests/PriceTest.php:27\n\n"
     "FAILURES!\nTests: 13, Assertions: 32, Failures: 1."),
    ("dotnet_pass", True, "dotnet test",
     "  Determining projects to restore...\n  All projects are up-to-date for restore.\n"
     "  Billing.Tests -> /src/Billing.Tests/bin/Debug/net8.0/Billing.Tests.dll\n"
     "Test run for /src/Billing.Tests/bin/Debug/net8.0/Billing.Tests.dll (.NETCoreApp,Version=v8.0)\n"
     "Starting test execution, please wait...\nA total of 1 test files matched the specified pattern.\n\n"
     "Passed!  - Failed:     0, Passed:    27, Skipped:     0, Total:    27, Duration: 61 ms - Billing.Tests.dll (net8.0)"),
    ("dotnet_fail", False, "dotnet test",
     "  Failed Billing.Tests.InvoiceTests.Total_IncludesTax [4 ms]\n  Error Message:\n"
     "   Assert.Equal() Failure: Values differ\nExpected: 107.5\nActual:   100\n  Stack Trace:\n"
     "     at Billing.Tests.InvoiceTests.Total_IncludesTax() in /src/Billing.Tests/InvoiceTests.cs:line 19\n\n"
     "Failed!  - Failed:     1, Passed:    26, Skipped:     0, Total:    27, Duration: 63 ms - Billing.Tests.dll (net8.0)"),
    ("ctest_pass", True, "ctest --test-dir build",
     "Test project /home/dev/geo/build\n    Start 1: vec_tests\n1/3 Test #1: vec_tests ........................   Passed    0.01 sec\n"
     "    Start 2: poly_tests\n2/3 Test #2: poly_tests .......................   Passed    0.02 sec\n"
     "    Start 3: io_tests\n3/3 Test #3: io_tests .........................   Passed    0.01 sec\n\n"
     "100% tests passed, 0 tests failed out of 3\n\nTotal Test time (real) =   0.05 sec"),
    ("ctest_fail", False, "ctest --test-dir build",
     "Test project /home/dev/geo/build\n    Start 1: vec_tests\n1/3 Test #1: vec_tests ........................   Passed    0.01 sec\n"
     "    Start 2: poly_tests\n2/3 Test #2: poly_tests .......................***Failed    0.02 sec\n"
     "    Start 3: io_tests\n3/3 Test #3: io_tests .........................   Passed    0.01 sec\n\n"
     "67% tests passed, 1 tests failed out of 3\n\nTotal Test time (real) =   0.06 sec\n\n"
     "The following tests FAILED:\n\t  2 - poly_tests (Failed)\nErrors while running CTest"),
    ("jest_coverage_pass", True, "npx jest --coverage",
     "PASS  src/cart.test.js\nPASS  src/price.test.js\n"
     "----------|---------|----------|---------|---------|-------------------\n"
     "File      | % Stmts | % Branch | % Funcs | % Lines | Uncovered Line #s \n"
     "----------|---------|----------|---------|---------|-------------------\n"
     "All files |   94.11 |    83.33 |     100 |   94.11 |                   \n"
     " cart.js  |   91.66 |       75 |     100 |   91.66 | 14                \n"
     " price.js |     100 |      100 |     100 |     100 |                   \n"
     "----------|---------|----------|---------|---------|-------------------\n\n"
     "Test Suites: 2 passed, 2 total\nTests:       18 passed, 18 total\nSnapshots:   0 total\nTime:        0.912 s\n"
     "Ran all test suites."),
    ("agent_en_pass", True, None, "Ran the full suite after the change: 214 passed, 0 failed."),
    ("agent_en_fail", False, None, "I fixed the bug and reran the tests; 2 are still failing in test_auth.py."),
    ("agent_zh_pass", True, None, "已修复，重新跑了一遍测试：全部 58 个用例通过。"),
    ("agent_zh_fail", False, None, "修改后重新跑了测试，还有 3 个失败，都在 test_order.py 里。"),
]


def state(command, output):
    return output if command is None else {"command": command, "output": output}
