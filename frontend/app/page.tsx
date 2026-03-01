import { URDFViewer } from "./components/URDFViewer";

export default function Home() {
  return (
    <div className="flex flex-col h-screen w-screen">
      <header className="bg-white dark:bg-black border-b p-4">
        <h1 className="text-2xl font-bold">ANYmal Robot Visualization</h1>
      </header>
      <div className="flex-1" style={{ minHeight: 0 }}>
        <URDFViewer />
      </div>
    </div>
  );
}
