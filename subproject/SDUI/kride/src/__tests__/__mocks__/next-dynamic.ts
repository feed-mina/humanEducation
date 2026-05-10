const dynamic = (_fn: any, _opts?: any) => {
  const Comp = () => null;
  Comp.displayName = "DynamicMock";
  return Comp;
};
export default dynamic;
