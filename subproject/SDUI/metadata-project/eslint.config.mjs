import {defineConfig, globalIgnores} from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
    ...nextVitals,
    ...nextTs,
    {
        // 규칙을 완화하는 설정 추가
        rules: {
            "@typescript-eslint/no-explicit-any": "off",      // any 사용 허용
            "@typescript-eslint/no-unused-vars": "off",      // 안 쓰는 변수 허용
            "@typescript-eslint/no-unsafe-assignment": "off", // unsafe 할당 허용
            "@typescript-eslint/no-unsafe-member-access": "off", // unsafe 접근 허용
        },
    },
    globalIgnores([
        ".next/**",
        "out/**",
        "build/**",
        "next-env.d.ts",
    ]),
]);

export default eslintConfig;