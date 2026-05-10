export const useRouter = () => ({ push: jest.fn(), replace: jest.fn() });
export const usePathname = () => "/";
export const useSearchParams = () => new URLSearchParams();
