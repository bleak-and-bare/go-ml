import { createTheme } from "@mantine/core";

export const theme = createTheme({
    breakpoints: {
        xs: "360px",
        sm: "480px",
        md: "640px",
        lg: "768px",
        xl: "1024px",
    },
    headings: {
        sizes: {
            h1: { fontSize: "1.6rem" },
            h2: { fontSize: "1.4rem" },
            h3: { fontSize: "1.2rem" },
            h4: { fontSize: "1rem" },
        }
    }
    /* Put your mantine theme override here */
});
