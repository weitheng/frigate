import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { z } from "zod";

interface UploadImageDialogProps {
  open: boolean;
  title: string;
  description: string;
  setOpen: (open: boolean) => void;
  onSave: (file: File) => void;
}

// Define form schema
const formSchema = z.object({
  file: z.instanceof(FileList).refine((files) => files.length > 0, {
    message: "Please select an image file.",
  }),
});

type FormData = z.infer<typeof formSchema>;

export default function UploadImageDialog({
  open,
  title,
  description,
  setOpen,
  onSave,
}: UploadImageDialogProps) {
  const [isUploading, setIsUploading] = useState(false);

  const form = useForm<FormData>({
    resolver: zodResolver(formSchema),
  });

  const handleSubmit = async (data: FormData) => {
    if (!data.file?.[0]) {
      return;
    }

    try {
      setIsUploading(true);
      await onSave(data.file[0]);
      form.reset();
      setOpen(false);
      toast.success("File uploaded successfully");
    } catch (error) {
      toast.error(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="file"
              render={({ field: { onChange, ...field } }) => (
                <FormItem>
                  <FormControl>
                    <Input
                      type="file"
                      accept="image/*"
                      disabled={isUploading}
                      onChange={(e) => {
                        onChange(e.target.files);
                      }}
                      {...field}
                    />
                  </FormControl>
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button 
                type="submit" 
                disabled={isUploading}
              >
                {isUploading ? "Uploading..." : "Save"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
