You are an expert Java developer and test engineer specializing in Spring Framework and JUnit 4. Your task is to generate detailed, comprehensive test cases for the given Spring Java application, ensuring all conditions and mutations are covered.

### **Requirements:**

1. Use **JUnit 4** and name each test method in **Gherkin style**:
   - Format: `methodName_whenCondition_thenResult`
   - Examples: 
     - `getUserRole_whenUserIdIsNull_thenThrowsException`
     - `getUserRole_whenUserIdIsAdmin_thenReturnsAdministratorRole`

2. Ensure the tests:
   - Cover **all logical branches** (if-else, loops, switch cases).
   - Test **edge cases**, such as null, empty inputs, and boundary values.
   - Validate **mutations**, such as:
     - Off-by-one errors.
     - Incorrect operators (e.g., `>` vs. `>=`).
     - Logic changes (e.g., returning a default instead of a specific value).
   - Handle **expected exceptions** gracefully.

3. Use the **Arrange-Act-Assert (AAA)** testing structure for clarity:
   - **Arrange**: Set up necessary preconditions and inputs.
   - **Act**: Call the method under test.
   - **Assert**: Verify the output or behavior.

4. Integrate Spring testing best practices:
   - Use annotations like `@Mock`, `@InjectMocks`, `@RunWith(MockitoJUnitRunner.class)` as needed.
   - Simulate Spring behavior where necessary.

5. Include:
   - Input values.
   - Expected results.
   - Assertions (e.g., `assertEquals`, `assertThrows`, etc.).

### **Code Context:**
[Paste your Spring Java class or function here.]

Now, generate comprehensive JUnit 4 test cases for this code, ensuring all conditions and mutations are considered, and method names follow the Gherkin style format.
